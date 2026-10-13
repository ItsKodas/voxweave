mod setup;

use std::ffi::OsString;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use tauri::menu::{Menu, MenuItem};
use tauri::tray::TrayIconBuilder;
use tauri::{AppHandle, Emitter, Manager, RunEvent, State};
use tauri_plugin_global_shortcut::{Code, GlobalShortcutExt, Modifiers, Shortcut, ShortcutState};

struct Backend {
    port: u16,
}

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .and_then(|l| l.local_addr())
        .map(|a| a.port())
        .expect("no free port")
}

fn backend_dir() -> PathBuf {
    match std::env::var("AIVOICE_BACKEND_DIR") {
        Ok(dir) => PathBuf::from(dir),
        Err(_) => PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("..").join("backend"),
    }
}

const MAX_LOG_BYTES: u64 = 5 * 1024 * 1024;

/// Where the backend's output goes: `data/backend.log` next to the backend directory.
fn log_path(backend_dir: &Path) -> PathBuf {
    backend_dir.join("..").join("data").join("backend.log")
}

/// Opens the log for appending, creating its directory. A log larger than `max_bytes` is started afresh.
fn open_log(path: &Path, max_bytes: u64) -> std::io::Result<File> {
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir)?;
    }
    if fs::metadata(path).map(|meta| meta.len() > max_bytes).unwrap_or(false) {
        File::create(path)?;
    }
    OpenOptions::new().create(true).append(true).open(path)
}

/// How the backend is started. The source checkout and the installed app differ only in these values.
#[derive(Debug, Clone, PartialEq)]
struct Launch {
    python: PathBuf,
    dir: PathBuf,
    log: PathBuf,
    /// Variables of the shell's own environment the backend must not see, and the ones it is given.
    drop_env: Vec<OsString>,
    env: Vec<(&'static str, OsString)>,
}

/// The source checkout: the backend folder fixed at compile time, its own `.venv`, nothing added to the
/// environment, the log in `data/` next to it.
fn dev_launch(dir: PathBuf) -> Launch {
    Launch {
        python: dir.join(".venv").join("Scripts").join("python.exe"),
        log: log_path(&dir),
        dir,
        drop_env: Vec::new(),
        env: Vec::new(),
    }
}

/// The installed app: the environment the first-run setup built, the bundled source on PYTHONPATH, and
/// everything written under the root. The log is `<data>\backend.log`, as in the source checkout.
fn installed_launch(layout: &setup::Installed, inherited: impl IntoIterator<Item = OsString>) -> Launch {
    Launch {
        python: layout.python(),
        dir: layout.root.clone(),
        log: layout.backend_log(),
        drop_env: setup::inherited_to_drop(inherited),
        env: layout.backend_env(),
    }
}

fn spawn_backend(port: u16, launch: &Launch) -> std::io::Result<Child> {
    let mut cmd = Command::new(&launch.python);
    cmd.current_dir(&launch.dir).args([
        "-m",
        "aivoice",
        "--port",
        &port.to_string(),
        "--parent-pid",
        &std::process::id().to_string(),
    ]);
    for name in &launch.drop_env {
        cmd.env_remove(name);
    }
    for (name, value) in &launch.env {
        cmd.env(name, value);
    }
    // The backend has no console of its own, so its output (startup errors, tracebacks) goes to a log
    // file. Unbuffered, so the reason for a crash is on disk before the process is gone.
    match open_log(&launch.log, MAX_LOG_BYTES).and_then(|out| Ok((out.try_clone()?, out))) {
        Ok((out, err)) => {
            cmd.env("PYTHONUNBUFFERED", "1").stdout(Stdio::from(out)).stderr(Stdio::from(err));
        }
        Err(err) => eprintln!("could not open the backend log, its output is not saved: {err}"),
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    cmd.spawn()
}

const MIN_HEALTHY_RUN: Duration = Duration::from_secs(5);
const MIN_DELAY: Duration = Duration::from_secs(1);
const MAX_DELAY: Duration = Duration::from_secs(30);

/// Restart delay after a backend exit: a short run doubles the previous delay (capped), a long run resets it.
fn next_delay(previous: Duration, ran_for: Duration) -> Duration {
    if ran_for >= MIN_HEALTHY_RUN {
        MIN_DELAY
    } else {
        (previous * 2).clamp(MIN_DELAY, MAX_DELAY)
    }
}

/// Kills the backend and its descendants. Each backend is a venv launcher plus the real interpreter as its child.
fn kill_tree(child: &mut Child) {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        let _ = Command::new("taskkill")
            .args(["/PID", &child.id().to_string(), "/T", "/F"])
            .creation_flags(0x0800_0000) // CREATE_NO_WINDOW
            .status();
    }
    let _ = child.kill();
    let _ = child.wait();
}

/// Keeps the backend alive. A restarted backend has no live session, so live mode comes back stopped.
/// While `hold` is set (the installed app's setup is running) no backend runs: its environment is being built.
fn supervise(port: u16, launch: Launch, shell: Arc<Shell>) {
    let (child, quitting, hold) = (shell.backend.clone(), shell.quitting.clone(), shell.hold.clone());
    thread::spawn(move || {
        let mut delay = MIN_DELAY / 2;
        while !quitting.load(Ordering::SeqCst) {
            if let Some(mut old) = child.lock().unwrap().take() {
                kill_tree(&mut old);
            }
            if hold.load(Ordering::SeqCst) {
                delay = MIN_DELAY / 2;
                thread::sleep(Duration::from_millis(200));
                continue;
            }
            let started = Instant::now();
            match spawn_backend(port, &launch) {
                Ok(mut process) => {
                    let mut guard = child.lock().unwrap();
                    if quitting.load(Ordering::SeqCst) {
                        kill_tree(&mut process);
                        return;
                    }
                    if hold.load(Ordering::SeqCst) {
                        kill_tree(&mut process);
                        continue;
                    }
                    *guard = Some(process);
                }
                Err(err) => eprintln!("failed to start backend: {err}"),
            }
            loop {
                thread::sleep(Duration::from_millis(500));
                if quitting.load(Ordering::SeqCst) {
                    return;
                }
                if hold.load(Ordering::SeqCst) {
                    break;
                }
                let mut guard = child.lock().unwrap();
                let exited = match guard.as_mut() {
                    Some(process) => !matches!(process.try_wait(), Ok(None)),
                    None => true,
                };
                if exited {
                    break;
                }
            }
            if hold.load(Ordering::SeqCst) {
                continue;
            }
            delay = next_delay(delay, started.elapsed());
            thread::sleep(delay);
        }
    });
}

/// What the commands, the backend supervisor and the setup share.
struct Shell {
    /// None in the source checkout.
    installed: Option<setup::Installed>,
    version: String,
    state: Mutex<setup::SetupState>,
    backend: Arc<Mutex<Option<Child>>>,
    /// The running setup (uv, and below it the bootstrap and whatever that started).
    setup_child: Mutex<Option<Child>>,
    setup_running: AtomicBool,
    hold: Arc<AtomicBool>,
    quitting: Arc<AtomicBool>,
}

impl Shell {
    fn new(installed: Option<setup::Installed>, version: String) -> Self {
        let ready = installed.as_ref().map(|layout| layout.setup_complete(&version));
        let state = match (&installed, ready) {
            (Some(layout), Some(ready)) => setup::SetupState::installed(layout, ready),
            _ => setup::SetupState::dev(),
        };
        Shell {
            installed,
            version,
            state: Mutex::new(state),
            backend: Arc::new(Mutex::new(None)),
            setup_child: Mutex::new(None),
            setup_running: AtomicBool::new(false),
            hold: Arc::new(AtomicBool::new(ready == Some(false))),
            quitting: Arc::new(AtomicBool::new(false)),
        }
    }

    fn snapshot(&self) -> setup::SetupState {
        self.state.lock().unwrap().clone()
    }

    /// Changes the setup state and tells the window.
    fn update(&self, notify: &Notify, change: impl FnOnce(&mut setup::SetupState)) {
        let state = {
            let mut guard = self.state.lock().unwrap();
            change(&mut guard);
            guard.clone()
        };
        notify(state);
    }
}

/// How the window is told about a change of the setup: a `setup-progress` event in the app, a list in the tests.
type Notify = dyn Fn(setup::SetupState) + Send + Sync;

/// Starts the installed app's setup unless one is running. The backend is stopped first and stays stopped
/// until the setup has succeeded: it must not hold files of the environment that is being built or repaired.
fn begin_setup(app: &AppHandle, shell: &Arc<Shell>) {
    let Some(layout) = shell.installed.clone() else { return };
    if shell.setup_running.swap(true, Ordering::SeqCst) {
        return;
    }
    shell.hold.store(true, Ordering::SeqCst);
    let app = app.clone();
    let notify = move |state: setup::SetupState| {
        let _ = app.emit("setup-progress", state);
    };
    shell.update(&notify, |state| state.restart());
    let shell = shell.clone();
    thread::spawn(move || {
        let waited = Instant::now();
        while shell.backend.lock().unwrap().is_some() && waited.elapsed() < Duration::from_secs(10) {
            thread::sleep(Duration::from_millis(100)); // the supervisor ends the backend when it sees the hold
        }
        match run_setup(&shell, &layout, &notify) {
            Ok(()) => {
                shell.update(&notify, |state| state.succeed());
                shell.hold.store(false, Ordering::SeqCst);
            }
            Err(error) => shell.update(&notify, |state| state.fail(error)),
        }
        shell.setup_running.store(false, Ordering::SeqCst);
    });
}

const OUTPUT_TAIL_LINES: usize = 4;

/// Runs the bundled bootstrap with the bundled uv. Ok only when it reported success and what it made is
/// really there.
fn run_setup(shell: &Shell, layout: &setup::Installed, notify: &Notify) -> Result<(), String> {
    fs::create_dir_all(layout.runtime())
        .map_err(|err| format!("The folder {} could not be created: {err}", layout.runtime().display()))?;
    let (program, args) = layout.setup_command(&shell.version);
    let mut cmd = Command::new(&program);
    cmd.args(&args).current_dir(&layout.root);
    for name in setup::inherited_to_drop(std::env::vars_os().map(|(name, _)| name)) {
        cmd.env_remove(name);
    }
    for (name, value) in layout.setup_env() {
        cmd.env(name, value);
    }
    follow_setup(cmd, shell, layout, notify)
}

/// Starts the setup process and follows it to its end: its progress lines become the state the window
/// shows, what it prints as errors goes to the setup log, and its last line decides the result.
fn follow_setup(mut cmd: Command, shell: &Shell, layout: &setup::Installed, notify: &Notify) -> Result<(), String> {
    let program = cmd.get_program().to_string_lossy().into_owned();
    cmd.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    let mut child = cmd.spawn().map_err(|err| format!("The setup could not be started ({program}): {err}"))?;
    let (stdout, stderr) = (child.stdout.take(), child.stderr.take());
    *shell.setup_child.lock().unwrap() = Some(child);
    if shell.quitting.load(Ordering::SeqCst) {
        if let Some(mut process) = shell.setup_child.lock().unwrap().take() {
            kill_tree(&mut process);
        }
        return Err("The app is closing".into());
    }

    // What uv itself prints (the Python download, its own errors) goes to the setup log; the last lines
    // are kept for the error message of a setup that never got as far as the bootstrap.
    let tail = Arc::new(Mutex::new(Vec::<String>::new()));
    let log_file = open_log(&layout.setup_log(), MAX_LOG_BYTES).ok();
    let errors = stderr.map(|stream| {
        let tail = tail.clone();
        thread::spawn(move || {
            let mut log_file = log_file;
            for line in BufReader::new(stream).split(b'\n').map_while(Result::ok) {
                let line = String::from_utf8_lossy(&line).trim_end().to_string();
                if let Some(file) = log_file.as_mut() {
                    let _ = writeln!(file, "{line}");
                }
                if !line.trim().is_empty() {
                    let mut kept = tail.lock().unwrap();
                    kept.push(line);
                    if kept.len() > OUTPUT_TAIL_LINES {
                        kept.remove(0);
                    }
                }
            }
        })
    });

    let mut started = false;
    let mut done: Option<(bool, Option<String>)> = None;
    if let Some(stream) = stdout {
        for line in BufReader::new(stream).split(b'\n').map_while(Result::ok) {
            match setup::parse_line(&String::from_utf8_lossy(&line)) {
                Some(setup::Line::Progress { step, percent, message }) => {
                    started = true;
                    shell.update(notify, |state| state.progress(step, percent, message));
                }
                Some(setup::Line::Done { ok, error }) => {
                    started = true;
                    done = Some((ok, error));
                }
                None => {}
            }
        }
    }
    if let Some(thread) = errors {
        let _ = thread.join();
    }
    // The pipes are closed, so the process is ending; a quit may have taken and killed it meanwhile.
    let exit = shell.setup_child.lock().unwrap().take().and_then(|mut process| process.wait().ok()?.code());
    if shell.quitting.load(Ordering::SeqCst) {
        return Err("The app is closing".into());
    }
    match done {
        Some((true, _)) if layout.setup_complete(&shell.version) => Ok(()),
        Some((true, _)) => Err(format!(
            "The setup finished but its files are incomplete. Press Retry. The details are in {}.",
            layout.setup_log().display()
        )),
        other => Err(setup::failure_message(
            other.and_then(|(_, error)| error),
            started,
            exit,
            &tail.lock().unwrap(),
            &layout.setup_log(),
        )),
    }
}

#[tauri::command]
fn backend_port(state: State<Backend>) -> u16 {
    state.port
}

/// Whether this is the installed app, and how far its first-run setup is.
#[tauri::command]
fn setup_state(shell: State<Arc<Shell>>) -> setup::SetupState {
    shell.snapshot()
}

/// Starts the setup again: Retry after a failure, or a repair asked for by the user. Does nothing in the
/// source checkout or while a setup is running.
#[tauri::command]
fn start_setup(app: AppHandle, shell: State<Arc<Shell>>) -> setup::SetupState {
    begin_setup(&app, &shell);
    shell.snapshot()
}

/// The installed app's folders, or None for the source checkout. Decided once, when the app starts.
fn installed_layout(context: &tauri::Context) -> Option<setup::Installed> {
    let resources = tauri::utils::platform::resource_dir(context.package_info(), &tauri::utils::Env::default()).ok();
    let local_data = std::env::var_os("LOCALAPPDATA").map(PathBuf::from);
    setup::detect(tauri::is_dev(), resources.as_deref(), local_data.as_deref(), std::env::var_os(setup::ROOT_OVERRIDE))
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let port = free_port();
    let context = tauri::generate_context!();
    let installed = installed_layout(&context);
    let launch = match &installed {
        Some(layout) => installed_launch(layout, std::env::vars_os().map(|(name, _)| name)),
        None => dev_launch(backend_dir()),
    };
    let shell = Arc::new(Shell::new(installed, context.package_info().version.to_string()));
    let (setup_shell, exit_shell) = (shell.clone(), shell.clone());

    let app = tauri::Builder::default()
        .plugin(
            tauri_plugin_global_shortcut::Builder::new()
                .with_handler(|app, _shortcut, event| {
                    if event.state() == ShortcutState::Pressed {
                        let _ = app.emit("live-toggle", ());
                    }
                })
                .build(),
        )
        .manage(Backend { port })
        .manage(shell)
        .invoke_handler(tauri::generate_handler![backend_port, setup_state, start_setup])
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                // Live mode keeps running in the tray; "Quit" in the tray menu exits.
                api.prevent_close();
                let _ = window.hide();
            }
        })
        .setup(move |app| {
            if let Err(err) = app
                .global_shortcut()
                .register(Shortcut::new(Some(Modifiers::CONTROL | Modifiers::ALT), Code::KeyV))
            {
                eprintln!("could not register Ctrl+Alt+V: {err}");
            }

            let converted = MenuItem::with_id(app, "converted", "Converted", true, None::<&str>)?;
            let muted = MenuItem::with_id(app, "muted", "Muted", true, None::<&str>)?;
            let off = MenuItem::with_id(app, "off", "Off", true, None::<&str>)?;
            let show = MenuItem::with_id(app, "show", "Show window", true, None::<&str>)?;
            let quit = MenuItem::with_id(app, "quit", "Quit", true, None::<&str>)?;
            let menu = Menu::with_items(app, &[&converted, &muted, &off, &show, &quit])?;
            TrayIconBuilder::new()
                .icon(app.default_window_icon().expect("window icon").clone())
                .tooltip("Voxweave")
                .menu(&menu)
                .on_menu_event(|app, event| match event.id.as_ref() {
                    "quit" => app.exit(0),
                    "show" => {
                        if let Some(window) = app.get_webview_window("main") {
                            let _ = window.show();
                            let _ = window.set_focus();
                        }
                    }
                    mode => {
                        let _ = app.emit("live-mode", mode.to_string());
                    }
                })
                .build(app)?;
            supervise(port, launch, setup_shell.clone());
            if setup_shell.hold.load(Ordering::SeqCst) {
                // The installed app's first start, an update, or a damaged environment.
                begin_setup(app.handle(), &setup_shell);
            }
            Ok(())
        })
        .build(context)
        .expect("error while building tauri application");

    app.run(move |_handle, event| {
        if let RunEvent::Exit = event {
            exit_shell.quitting.store(true, Ordering::SeqCst);
            if let Some(mut process) = exit_shell.backend.lock().unwrap().take() {
                kill_tree(&mut process);
            }
            // A setup in progress is ended with everything it started; the next start completes it.
            if let Some(mut process) = exit_shell.setup_child.lock().unwrap().take() {
                kill_tree(&mut process);
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn free_port_is_bindable() {
        let port = free_port();
        assert!(port > 0);
        assert!(TcpListener::bind(("127.0.0.1", port)).is_ok());
    }

    #[test]
    fn next_delay_doubles_on_short_runs_and_caps() {
        let short = Duration::from_secs(1);
        assert_eq!(next_delay(Duration::from_secs(1), short), Duration::from_secs(2));
        assert_eq!(next_delay(Duration::from_secs(2), short), Duration::from_secs(4));
        assert_eq!(next_delay(Duration::from_secs(20), short), Duration::from_secs(30));
        assert_eq!(next_delay(Duration::from_secs(30), short), Duration::from_secs(30));
    }

    #[test]
    fn next_delay_resets_after_a_long_run() {
        assert_eq!(next_delay(Duration::from_secs(30), Duration::from_secs(5)), Duration::from_secs(1));
    }

    #[test]
    fn log_path_is_data_backend_log_next_to_the_backend_dir() {
        assert_eq!(
            log_path(Path::new("C:/x/backend")),
            PathBuf::from("C:/x/backend").join("..").join("data").join("backend.log")
        );
    }

    fn scratch_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("aivoice-test-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn open_log_creates_the_directory_and_appends() {
        use std::io::Write;
        let dir = scratch_dir("append");
        let path = log_path(&dir.join("backend"));
        for line in ["first\n", "second\n"] {
            let mut log = open_log(&path, MAX_LOG_BYTES).expect("log opens");
            log.write_all(line.as_bytes()).unwrap();
        }
        assert_eq!(std::fs::read_to_string(dir.join("data").join("backend.log")).unwrap(), "first\nsecond\n");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn open_log_starts_afresh_only_when_over_the_size_limit() {
        use std::io::Write;
        let dir = scratch_dir("truncate");
        let path = dir.join("data").join("backend.log");
        open_log(&path, 10).unwrap().write_all(b"0123456789").unwrap();
        open_log(&path, 10).unwrap().write_all(b"ab").unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "0123456789ab");
        open_log(&path, 10).unwrap().write_all(b"new").unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "new");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn open_log_reports_failure_instead_of_panicking() {
        let dir = scratch_dir("blocked");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("data"), b"a file where the directory should be").unwrap();
        assert!(open_log(&dir.join("data").join("backend.log"), MAX_LOG_BYTES).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_source_checkout_starts_the_backend_exactly_as_before() {
        let launch = dev_launch(PathBuf::from("C:/x/backend"));
        assert_eq!(launch.python, PathBuf::from("C:/x/backend").join(".venv").join("Scripts").join("python.exe"));
        assert_eq!(launch.dir, PathBuf::from("C:/x/backend"));
        assert_eq!(launch.log, log_path(Path::new("C:/x/backend")));
        assert!(launch.env.is_empty() && launch.drop_env.is_empty(), "nothing is added to or taken from the environment");
    }

    #[test]
    fn the_installed_app_starts_the_backend_from_its_own_environment() {
        let layout = setup::Installed { root: PathBuf::from(r"C:\L\Voxweave"), resources: PathBuf::from(r"C:\P\Voxweave") };
        let launch = installed_launch(&layout, ["PATH", "PYTHONHOME", "UV_CACHE_DIR"].map(OsString::from));
        assert_eq!(launch.python, PathBuf::from(r"C:\L\Voxweave\runtime\backend\.venv\Scripts\python.exe"));
        assert_eq!(launch.dir, PathBuf::from(r"C:\L\Voxweave"));
        assert_eq!(launch.log, PathBuf::from(r"C:\L\Voxweave\data\backend.log"));
        assert_eq!(launch.drop_env, ["PYTHONHOME", "UV_CACHE_DIR"].map(OsString::from));
        let env: Vec<(&str, String)> = launch.env.iter().map(|(k, v)| (*k, v.to_string_lossy().into_owned())).collect();
        assert!(env.contains(&("PYTHONPATH", r"C:\P\Voxweave\backend".to_string())));
        assert!(env.contains(&("AIVOICE_DATA", r"C:\L\Voxweave\data".to_string())));
        assert!(env.contains(&("AIVOICE_INSTALLED", "1".to_string())));
    }

    #[test]
    fn the_source_checkout_has_no_setup_and_never_holds_the_backend() {
        let shell = Shell::new(None, "0.1.0".into());
        assert_eq!(shell.snapshot(), setup::SetupState::dev());
        assert!(!shell.hold.load(Ordering::SeqCst));
    }

    #[test]
    fn an_installed_app_without_a_finished_setup_holds_the_backend() {
        let dir = scratch_dir("shell-installed");
        let layout = setup::Installed { root: dir.join("root"), resources: dir.join("program") };
        let shell = Shell::new(Some(layout.clone()), "0.1.0".into());
        assert_eq!(shell.snapshot().status, "running");
        assert!(shell.hold.load(Ordering::SeqCst));

        fs::create_dir_all(layout.python().parent().unwrap()).unwrap();
        fs::write(layout.python(), b"").unwrap();
        fs::create_dir_all(layout.base_model().parent().unwrap()).unwrap();
        fs::write(layout.base_model(), b"").unwrap();
        fs::write(layout.record(), br#"{"version": "0.1.0"}"#).unwrap();
        let shell = Shell::new(Some(layout), "0.1.0".into());
        assert_eq!(shell.snapshot().status, "ready");
        assert!(!shell.hold.load(Ordering::SeqCst));
        let _ = fs::remove_dir_all(&dir);
    }

    /// A shell for an installed app in a scratch folder, and a stand-in for uv: a batch file that prints
    /// what the test wants the bootstrap to have printed.
    fn setup_rig(name: &str, script: &str) -> (PathBuf, setup::Installed, Shell, Command) {
        let dir = scratch_dir(name);
        let layout = setup::Installed { root: dir.join("root"), resources: dir.join("program") };
        fs::create_dir_all(layout.runtime()).unwrap();
        let batch = dir.join("fake-setup.cmd");
        fs::write(&batch, format!("@echo off\r\n{}\r\n", script.replace('\n', "\r\n"))).unwrap();
        let mut cmd = Command::new("cmd");
        cmd.arg("/C").arg(&batch).current_dir(&dir);
        let shell = Shell::new(Some(layout.clone()), "0.1.0".into());
        (dir, layout, shell, cmd)
    }

    fn finish_files(layout: &setup::Installed) {
        fs::create_dir_all(layout.python().parent().unwrap()).unwrap();
        fs::write(layout.python(), b"").unwrap();
        fs::create_dir_all(layout.base_model().parent().unwrap()).unwrap();
        fs::write(layout.base_model(), b"").unwrap();
        fs::write(layout.record(), br#"{"version": "0.1.0"}"#).unwrap();
    }

    type Seen = Arc<Mutex<Vec<setup::SetupState>>>;

    fn recorder() -> (Seen, Box<Notify>) {
        let seen: Seen = Arc::new(Mutex::new(Vec::new()));
        let sink = seen.clone();
        (seen, Box::new(move |state| sink.lock().unwrap().push(state)))
    }

    const PROGRESS: &str = r#"echo Downloading cpython 1>&2
echo {"step": "packages", "percent": 40, "message": "Installing"}
echo not a json line
echo {"step": "model", "percent": 80, "message": "Downloading the model"}"#;

    #[test]
    fn a_setup_process_that_succeeds_is_followed_line_by_line() {
        let script = format!("{PROGRESS}\necho {{\"done\": true, \"ok\": true, \"error\": null}}");
        let (dir, layout, shell, cmd) = setup_rig("follow-ok", &script);
        finish_files(&layout);
        let (seen, notify) = recorder();
        assert_eq!(follow_setup(cmd, &shell, &layout, &notify), Ok(()));
        let seen = seen.lock().unwrap();
        let steps: Vec<(&str, u8, &str)> = seen.iter().map(|s| (s.step.as_str(), s.percent, s.message.as_str())).collect();
        assert_eq!(steps, [("packages", 40, "Installing"), ("model", 80, "Downloading the model")]);
        assert!(seen.iter().all(|s| s.status == "running"), "ready is only said once the caller has checked the result");
        assert!(shell.setup_child.lock().unwrap().is_none());
        assert!(fs::read_to_string(layout.setup_log()).unwrap().contains("Downloading cpython"));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_setup_that_reports_success_without_its_files_is_not_believed() {
        let (dir, layout, shell, cmd) = setup_rig("follow-hollow", r#"echo {"done": true, "ok": true, "error": null}"#);
        let (_seen, notify) = recorder();
        let error = follow_setup(cmd, &shell, &layout, &notify).unwrap_err();
        assert!(error.contains("incomplete") && error.contains("Retry"), "{error}");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_setup_that_fails_passes_on_the_reason_it_gave() {
        let script = format!("{PROGRESS}\necho {{\"done\": true, \"ok\": false, \"error\": \"No free space\"}}\nexit /b 1");
        let (dir, layout, shell, cmd) = setup_rig("follow-failed", &script);
        let (seen, notify) = recorder();
        assert_eq!(follow_setup(cmd, &shell, &layout, &notify), Err("No free space".into()));
        assert_eq!(seen.lock().unwrap().len(), 2);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_setup_that_dies_before_the_bootstrap_ran_is_explained_with_its_last_output() {
        let (dir, layout, shell, cmd) = setup_rig("follow-early", "echo error: failed to download python 1>&2\nexit /b 2");
        let (seen, notify) = recorder();
        let error = follow_setup(cmd, &shell, &layout, &notify).unwrap_err();
        assert!(error.contains("Python could not be downloaded"), "{error}");
        assert!(error.contains("failed to download python"), "{error}");
        assert!(seen.lock().unwrap().is_empty());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_setup_that_stops_halfway_without_a_last_line_names_the_exit_code_and_the_log() {
        let (dir, layout, shell, cmd) = setup_rig("follow-crash", &format!("{PROGRESS}\nexit /b 7"));
        let (_seen, notify) = recorder();
        let error = follow_setup(cmd, &shell, &layout, &notify).unwrap_err();
        assert!(error.contains("exit code 7") && error.contains("setup.log"), "{error}");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_setup_program_that_is_missing_is_a_readable_error() {
        let (dir, layout, shell, _cmd) = setup_rig("follow-missing", "");
        let (_seen, notify) = recorder();
        let error = follow_setup(Command::new(dir.join("no-such-uv.exe")), &shell, &layout, &notify).unwrap_err();
        assert!(error.contains("could not be started") && error.contains("no-such-uv.exe"), "{error}");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_setup_is_killed_with_what_it_started_when_the_app_quits() {
        // Stands in for a long download: prints one line, then waits far longer than the test runs.
        let script = "echo {\"step\": \"model\", \"percent\": 70, \"message\": \"Downloading\"}\nping -n 60 127.0.0.1 > nul";
        let (dir, layout, shell, cmd) = setup_rig("follow-quit", script);
        let shell = Arc::new(shell);
        let (seen, notify) = recorder();
        let worker = {
            let (shell, layout) = (shell.clone(), layout.clone());
            thread::spawn(move || follow_setup(cmd, &shell, &layout, &notify))
        };
        let waited = Instant::now();
        while seen.lock().unwrap().is_empty() && waited.elapsed() < Duration::from_secs(20) {
            thread::sleep(Duration::from_millis(50));
        }
        assert_eq!(seen.lock().unwrap().len(), 1, "the setup got as far as its first line");
        // What the Exit handler does.
        shell.quitting.store(true, Ordering::SeqCst);
        let mut process = shell.setup_child.lock().unwrap().take().expect("the running setup is known to the shell");
        let started = Instant::now();
        kill_tree(&mut process);
        assert_eq!(worker.join().unwrap(), Err("The app is closing".into()));
        assert!(started.elapsed() < Duration::from_secs(15), "the ping it started was ended too, not waited for");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn backend_dir_honours_env() {
        // SAFETY: only this test touches this variable.
        unsafe { std::env::set_var("AIVOICE_BACKEND_DIR", "C:/x/backend") };
        assert_eq!(backend_dir(), PathBuf::from("C:/x/backend"));
        unsafe { std::env::remove_var("AIVOICE_BACKEND_DIR") };
    }
}
