use std::fs::{self, File, OpenOptions};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use tauri::menu::{Menu, MenuItem};
use tauri::tray::TrayIconBuilder;
use tauri::{Emitter, Manager, RunEvent, State};
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

fn spawn_backend(port: u16) -> std::io::Result<Child> {
    let dir = backend_dir();
    let mut cmd = Command::new(dir.join(".venv").join("Scripts").join("python.exe"));
    cmd.current_dir(&dir).args([
        "-m",
        "aivoice",
        "--port",
        &port.to_string(),
        "--parent-pid",
        &std::process::id().to_string(),
    ]);
    // The backend has no console of its own, so its output (startup errors, tracebacks) goes to a log
    // file. Unbuffered, so the reason for a crash is on disk before the process is gone.
    match open_log(&log_path(&dir), MAX_LOG_BYTES).and_then(|out| Ok((out.try_clone()?, out))) {
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
fn supervise(port: u16, child: Arc<Mutex<Option<Child>>>, quitting: Arc<AtomicBool>) {
    thread::spawn(move || {
        let mut delay = MIN_DELAY / 2;
        while !quitting.load(Ordering::SeqCst) {
            if let Some(mut old) = child.lock().unwrap().take() {
                kill_tree(&mut old);
            }
            let started = Instant::now();
            match spawn_backend(port) {
                Ok(mut process) => {
                    let mut guard = child.lock().unwrap();
                    if quitting.load(Ordering::SeqCst) {
                        kill_tree(&mut process);
                        return;
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
                let mut guard = child.lock().unwrap();
                let exited = match guard.as_mut() {
                    Some(process) => !matches!(process.try_wait(), Ok(None)),
                    None => true,
                };
                if exited {
                    break;
                }
            }
            delay = next_delay(delay, started.elapsed());
            thread::sleep(delay);
        }
    });
}

#[tauri::command]
fn backend_port(state: State<Backend>) -> u16 {
    state.port
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let port = free_port();
    let child = Arc::new(Mutex::new(None));
    let quitting = Arc::new(AtomicBool::new(false));
    let (setup_child, setup_quitting) = (child.clone(), quitting.clone());

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
        .invoke_handler(tauri::generate_handler![backend_port])
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
                .tooltip("AI Voice")
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
            supervise(port, setup_child, setup_quitting);
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building tauri application");

    app.run(move |_handle, event| {
        if let RunEvent::Exit = event {
            quitting.store(true, Ordering::SeqCst);
            if let Some(mut process) = child.lock().unwrap().take() {
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
    fn backend_dir_honours_env() {
        // SAFETY: only this test touches this variable.
        unsafe { std::env::set_var("AIVOICE_BACKEND_DIR", "C:/x/backend") };
        assert_eq!(backend_dir(), PathBuf::from("C:/x/backend"));
        unsafe { std::env::remove_var("AIVOICE_BACKEND_DIR") };
    }
}
