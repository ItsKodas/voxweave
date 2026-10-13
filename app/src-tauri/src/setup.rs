//! The installed app: where its files live, whether the first-run setup is still needed, and the
//! command that runs it. Nothing here starts a process or touches Tauri, so all of it is unit tested.
//!
//! Installed (put on the computer by the installer):
//!
//! ```text
//! <resources>\backend\aivoice\...        the voice engine's source, read only
//! <resources>\setup\bootstrap.py         the first-run setup
//! <resources>\setup\requirements.lock    the pinned packages it installs
//! <resources>\bin\uv.exe                 runs the setup
//! <root>\runtime\backend\.venv           the Python environment the setup builds
//! <root>\runtime\setup.json              written last by a setup that succeeded
//! <root>\data                            voices, settings, logs (AIVOICE_DATA)
//! ```
//!
//! `<resources>` is the folder of the program and `<root>` is `%LOCALAPPDATA%\Voxweave`.

use std::ffi::OsString;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// The folder under the user's local application data that holds everything the installed app writes.
pub const ROOT_FOLDER: &str = "Voxweave";
pub const PYTHON_VERSION: &str = "3.12";
const BASE_MODEL: &str = "vec-768-layer-12.onnx";

/// Paths given by Windows can carry the `\\?\` prefix, which Python does not accept in PYTHONPATH.
pub fn plain(path: &Path) -> PathBuf {
    match path.to_str().and_then(|text| text.strip_prefix(r"\\?\")) {
        Some(rest) if !rest.starts_with("UNC\\") => PathBuf::from(rest),
        _ => path.to_path_buf(),
    }
}

/// The files the installer puts next to the program, relative to it. All of them must be there.
pub const BUNDLED_FILES: [&str; 4] =
    [r"backend\aivoice\__main__.py", r"setup\bootstrap.py", r"setup\requirements.lock", r"bin\uv.exe"];

/// The first of the installer's files that is not next to the program, if any.
pub fn missing_bundled_file(resources: &Path) -> Option<PathBuf> {
    BUNDLED_FILES.iter().map(|file| resources.join(file)).find(|path| !path.is_file())
}

#[derive(Debug, Clone, PartialEq)]
pub struct Installed {
    pub root: PathBuf,
    pub resources: PathBuf,
}

/// What kind of app this is. Decided once, when it starts.
#[derive(Debug, Clone, PartialEq)]
pub enum Mode {
    /// A development build (`npm run tauri dev`, `cargo test`): the source checkout, as it always was.
    Dev,
    /// The app the installer put on this computer, with all its files.
    Installed(Installed),
    /// A release build that cannot run: the message is shown and no backend is ever started. A release
    /// build never falls back to the source checkout's backend, whose path exists only on the PC it was
    /// built on; on any other PC that path is one somebody else could put a program at.
    Broken(String),
}

pub const MISSING_FILE: &str = "A file of the app is missing. Install Voxweave again.";
pub const NOT_AN_INSTALLER_BUILD: &str = "This copy of Voxweave was not built as an installer, so it has no voice \
    engine of its own. Build it with \"npm run installer\", or run it from the source checkout with \"npm run tauri dev\".";
pub const NO_LOCAL_APP_DATA: &str = "Windows did not say where this account's application data is kept \
    (LOCALAPPDATA), so Voxweave has nowhere to keep its files.";

/// `installer_build` is compiled in by `npm run installer` (scripts/build-installer.ps1 sets VOXWEAVE_INSTALLER).
/// Files lying next to the program do not decide the mode by themselves: Tauri copies bundle resources next
/// to every binary it builds and never removes them, so an earlier installer build leaves them behind for a
/// later plain build.
pub fn detect(dev_build: bool, installer_build: bool, resources: Option<&Path>, local_data: Option<&Path>) -> Mode {
    if dev_build {
        return Mode::Dev;
    }
    if !installer_build {
        return Mode::Broken(NOT_AN_INSTALLER_BUILD.into());
    }
    let Some(resources) = resources.map(plain) else { return Mode::Broken(MISSING_FILE.into()) };
    if missing_bundled_file(&resources).is_some() {
        return Mode::Broken(MISSING_FILE.into());
    }
    match local_data.map(plain) {
        Some(local) if local.is_absolute() => Mode::Installed(Installed { root: local.join(ROOT_FOLDER), resources }),
        _ => Mode::Broken(NO_LOCAL_APP_DATA.into()),
    }
}

/// A Windows tool by its full path under `%SystemRoot%\System32`. Never started by bare name: Windows looks
/// in the program's own folder first, and the installed app's folder can be written to by its user's programs.
pub fn system_tool(name: &str, system_root: Option<OsString>) -> PathBuf {
    let root = system_root.map(PathBuf::from).filter(|path| path.is_absolute()).unwrap_or_else(|| PathBuf::from(r"C:\Windows"));
    root.join("System32").join(name)
}

/// The part of `runtime\setup.json` the shell reads.
#[derive(Debug, Deserialize)]
struct SetupRecord {
    version: String,
}

impl Installed {
    pub fn runtime(&self) -> PathBuf {
        self.root.join("runtime")
    }

    pub fn data(&self) -> PathBuf {
        self.root.join("data")
    }

    pub fn backend_source(&self) -> PathBuf {
        self.resources.join("backend")
    }

    pub fn python(&self) -> PathBuf {
        self.runtime().join("backend").join(".venv").join("Scripts").join("python.exe")
    }

    pub fn base_model(&self) -> PathBuf {
        self.data().join("pretrained").join(BASE_MODEL)
    }

    pub fn record(&self) -> PathBuf {
        self.runtime().join("setup.json")
    }

    pub fn setup_log(&self) -> PathBuf {
        self.runtime().join("setup.log")
    }

    pub fn backend_log(&self) -> PathBuf {
        self.data().join("backend.log")
    }

    pub fn uv(&self) -> PathBuf {
        self.resources.join("bin").join("uv.exe")
    }

    pub fn bootstrap(&self) -> PathBuf {
        self.resources.join("setup").join("bootstrap.py")
    }

    /// True when a setup for exactly this app version finished and what it made is still there. Anything
    /// else (first start, an update, a deleted environment or model, an unreadable record) runs the setup,
    /// which completes or repairs whatever it finds.
    pub fn setup_complete(&self, version: &str) -> bool {
        let recorded = std::fs::read_to_string(self.record())
            .ok()
            .and_then(|text| serde_json::from_str::<SetupRecord>(&text).ok())
            .map(|record| record.version);
        recorded.as_deref() == Some(version) && self.python().is_file() && self.base_model().is_file()
    }

    /// The setup as an argument list: the bundled uv runs the bundled bootstrap with its own Python 3.12.
    pub fn setup_command(&self, version: &str) -> (PathBuf, Vec<OsString>) {
        let args: Vec<OsString> = vec![
            "run".into(),
            "--no-project".into(),
            "--python".into(),
            PYTHON_VERSION.into(),
            self.bootstrap().into(),
            "--root".into(),
            self.root.clone().into(),
            "--resources".into(),
            self.resources.clone().into(),
            "--app-version".into(),
            version.into(),
        ];
        (self.uv(), args)
    }

    /// What uv is told: its cache and its Python stay under the root, the user's own uv settings and
    /// Python installations play no part, and nothing is registered with Windows.
    pub fn setup_env(&self) -> Vec<(&'static str, OsString)> {
        vec![
            ("UV_CACHE_DIR", self.runtime().join("uv-cache").into()),
            ("UV_PYTHON_INSTALL_DIR", self.runtime().join("python").into()),
            ("UV_PYTHON_PREFERENCE", "only-managed".into()),
            ("UV_PYTHON_INSTALL_REGISTRY", "0".into()),
            ("UV_PYTHON_INSTALL_BIN", "0".into()),
            ("UV_NO_CONFIG", "1".into()),
            ("UV_NO_PROGRESS", "1".into()),
            ("PYTHONNOUSERSITE", "1".into()),
            ("PYTHONSAFEPATH", "1".into()),
        ]
    }

    /// What the voice engine is told: its source is the bundled folder (no editable install), its data
    /// folder, and that it is the installed app. Python writes no `__pycache__` into the program folder (the
    /// uninstaller would leave it behind); the packages were compiled by the setup, and the engine's own
    /// source compiles in a fraction of a second.
    pub fn backend_env(&self) -> Vec<(&'static str, OsString)> {
        vec![
            ("PYTHONPATH", self.backend_source().into()),
            ("AIVOICE_DATA", self.data().into()),
            ("AIVOICE_INSTALLED", "1".into()),
            ("PYTHONDONTWRITEBYTECODE", "1".into()),
            // Nothing from the user's own site-packages, and nothing from the folder the engine is started in.
            ("PYTHONNOUSERSITE", "1".into()),
            ("PYTHONSAFEPATH", "1".into()),
            ("HF_HOME", self.runtime().join("huggingface").into()),
        ]
    }
}

/// Variables of the user's own environment that must not reach uv or the voice engine's Python.
pub fn inherited_to_drop<I: IntoIterator<Item = OsString>>(names: I) -> Vec<OsString> {
    names
        .into_iter()
        .filter(|name| {
            let upper = name.to_string_lossy().to_uppercase();
            upper.starts_with("UV_") || matches!(upper.as_str(), "VIRTUAL_ENV" | "PYTHONPATH" | "PYTHONHOME")
        })
        .collect()
}

/// One line of the bootstrap's output.
#[derive(Debug, Clone, PartialEq)]
pub enum Line {
    Progress { step: String, percent: u8, message: String },
    Done { ok: bool, error: Option<String> },
}

#[derive(Deserialize)]
struct RawLine {
    done: Option<bool>,
    ok: Option<bool>,
    error: Option<String>,
    step: Option<String>,
    percent: Option<f64>,
    message: Option<String>,
}

/// Reads a line the bootstrap printed. Anything that is not one of its JSON lines is None.
pub fn parse_line(line: &str) -> Option<Line> {
    let raw: RawLine = serde_json::from_str(line.trim()).ok()?;
    if raw.done == Some(true) {
        return Some(Line::Done { ok: raw.ok == Some(true), error: raw.error });
    }
    Some(Line::Progress {
        step: raw.step?,
        percent: raw.percent?.clamp(0.0, 100.0) as u8,
        message: raw.message.unwrap_or_default(),
    })
}

/// What the window is told about the setup, by the `setup_state` command and by `setup-progress` events.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SetupState {
    /// "dev" (source checkout: there is no setup) or "installed".
    pub mode: &'static str,
    /// "ready", "running" or "failed".
    pub status: &'static str,
    pub step: String,
    pub percent: u8,
    pub message: String,
    pub error: Option<String>,
    /// No setup has ever finished here: this one downloads everything. False for the quick check after an
    /// update or a repair, which downloads only what is missing or changed.
    pub first_time: bool,
    /// Whether starting the setup again can help. False when the app itself is incomplete.
    pub can_retry: bool,
    /// Where the app keeps everything it writes; empty in dev mode.
    pub root: String,
    pub data_dir: String,
    pub setup_log: String,
}

pub const PYTHON_STEP_MESSAGE: &str = "Preparing Python";

impl SetupState {
    pub fn dev() -> Self {
        SetupState {
            mode: "dev",
            status: "ready",
            step: String::new(),
            percent: 100,
            message: String::new(),
            error: None,
            first_time: false,
            can_retry: false,
            root: String::new(),
            data_dir: String::new(),
            setup_log: String::new(),
        }
    }

    /// A release build that cannot run: the reason, and nothing to retry.
    pub fn broken(reason: &str) -> Self {
        SetupState { mode: "installed", status: "failed", percent: 0, error: Some(reason.into()), ..SetupState::dev() }
    }

    pub fn installed(layout: &Installed, ready: bool) -> Self {
        SetupState {
            mode: "installed",
            status: if ready { "ready" } else { "running" },
            step: if ready { String::new() } else { "python".into() },
            percent: if ready { 100 } else { 0 },
            message: if ready { String::new() } else { PYTHON_STEP_MESSAGE.into() },
            error: None,
            first_time: !layout.record().is_file(),
            can_retry: true,
            root: layout.root.display().to_string(),
            data_dir: layout.data().display().to_string(),
            setup_log: layout.setup_log().display().to_string(),
        }
    }

    /// A setup has been started (again): back to the first step, the old error gone.
    pub fn restart(&mut self, first_time: bool) {
        self.first_time = first_time;
        self.status = "running";
        self.step = "python".into();
        self.percent = 0;
        self.message = PYTHON_STEP_MESSAGE.into();
        self.error = None;
    }

    pub fn progress(&mut self, step: String, percent: u8, message: String) {
        self.step = step;
        self.percent = self.percent.max(percent);
        self.message = message;
    }

    pub fn succeed(&mut self) {
        self.status = "ready";
        self.percent = 100;
        self.error = None;
    }

    pub fn fail(&mut self, error: String) {
        self.status = "failed";
        self.error = Some(error);
    }
}

/// The message for a setup that ended without a usable result from the bootstrap.
pub fn failure_message(reported: Option<String>, started: bool, exit: Option<i32>, last_output: &[String], log: &Path) -> String {
    if let Some(error) = reported.filter(|text| !text.trim().is_empty()) {
        return error;
    }
    let tail = last_output.join(" | ");
    let detail = if tail.is_empty() { String::new() } else { format!(" Last output: {tail}") };
    if !started {
        return format!(
            "Python could not be downloaded, so the setup did not start. Check the internet connection and press Retry.{detail}"
        );
    }
    let code = exit.map(|code| format!("exit code {code}")).unwrap_or_else(|| "no exit code".into());
    format!("The setup stopped unexpectedly ({code}). The details are in {}.{detail}", log.display())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("voxweave-setup-test-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn bundle(dir: &Path) -> PathBuf {
        let resources = dir.join("program");
        for file in BUNDLED_FILES {
            let path = resources.join(file);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, b"").unwrap();
        }
        resources
    }

    fn layout() -> Installed {
        Installed { root: PathBuf::from(r"C:\Users\x\AppData\Local\Voxweave"), resources: PathBuf::from(r"C:\Program\Voxweave") }
    }

    const LOCAL: &str = r"C:\Users\x\AppData\Local";

    #[test]
    fn an_installer_build_with_all_its_files_is_installed_under_local_app_data() {
        let dir = scratch("installed");
        let resources = bundle(&dir);
        let Mode::Installed(found) = detect(false, true, Some(&resources), Some(Path::new(LOCAL))) else { panic!("not installed") };
        assert_eq!(found.root, PathBuf::from(r"C:\Users\x\AppData\Local\Voxweave"));
        assert_eq!(found.resources, resources);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_development_build_is_the_source_checkout_whatever_lies_beside_it() {
        let dir = scratch("dev");
        let resources = bundle(&dir);
        assert_eq!(detect(true, false, Some(&resources), Some(Path::new(LOCAL))), Mode::Dev);
        assert_eq!(detect(true, true, Some(&resources), Some(Path::new(LOCAL))), Mode::Dev);
        assert_eq!(detect(true, false, None, None), Mode::Dev);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_release_build_never_becomes_the_source_checkout() {
        let dir = scratch("never-dev");
        let resources = bundle(&dir);
        let local = Some(Path::new(LOCAL));
        for installer_build in [false, true] {
            for resources in [None, Some(dir.as_path()), Some(resources.as_path())] {
                for local in [None, local] {
                    assert_ne!(detect(false, installer_build, resources, local), Mode::Dev);
                }
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_installer_build_with_a_file_missing_says_so_and_is_not_installed() {
        let dir = scratch("missing");
        let local = Some(Path::new(LOCAL));
        assert_eq!(detect(false, true, None, local), Mode::Broken(MISSING_FILE.into()));
        assert_eq!(detect(false, true, Some(&dir), local), Mode::Broken(MISSING_FILE.into()));
        for file in BUNDLED_FILES {
            let resources = bundle(&dir);
            std::fs::remove_file(resources.join(file)).unwrap();
            assert_eq!(missing_bundled_file(&resources), Some(resources.join(file)));
            assert_eq!(detect(false, true, Some(&resources), local), Mode::Broken(MISSING_FILE.into()), "{file}");
        }
        assert_eq!(MISSING_FILE, "A file of the app is missing. Install Voxweave again.");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn files_left_by_an_earlier_installer_build_do_not_make_a_plain_release_build_installed() {
        let dir = scratch("stale");
        let resources = bundle(&dir);
        let found = detect(false, false, Some(&resources), Some(Path::new(LOCAL)));
        assert_eq!(found, Mode::Broken(NOT_AN_INSTALLER_BUILD.into()));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn without_a_usable_local_app_data_folder_nothing_is_started() {
        let dir = scratch("nolocal");
        let resources = bundle(&dir);
        assert_eq!(detect(false, true, Some(&resources), None), Mode::Broken(NO_LOCAL_APP_DATA.into()));
        // A relative value would put the root wherever the app happens to be started.
        assert_eq!(detect(false, true, Some(&resources), Some(Path::new("AppData"))), Mode::Broken(NO_LOCAL_APP_DATA.into()));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn system_tools_are_named_by_their_full_path() {
        assert_eq!(system_tool("taskkill.exe", Some(OsString::from(r"D:\Win"))), PathBuf::from(r"D:\Win\System32\taskkill.exe"));
        assert_eq!(system_tool("taskkill.exe", None), PathBuf::from(r"C:\Windows\System32\taskkill.exe"));
        assert_eq!(system_tool("cmd.exe", Some(OsString::from("Windows"))), PathBuf::from(r"C:\Windows\System32\cmd.exe"));
        assert!(system_tool("taskkill.exe", std::env::var_os("SystemRoot")).is_file());
    }

    #[test]
    fn the_long_path_prefix_is_removed() {
        assert_eq!(plain(Path::new(r"\\?\C:\a\b")), PathBuf::from(r"C:\a\b"));
        assert_eq!(plain(Path::new(r"C:\a\b")), PathBuf::from(r"C:\a\b"));
        assert_eq!(plain(Path::new(r"\\?\UNC\server\share")), PathBuf::from(r"\\?\UNC\server\share"));
    }

    #[test]
    fn everything_written_is_under_the_root_and_everything_read_under_the_resources() {
        let l = layout();
        assert_eq!(l.python(), PathBuf::from(r"C:\Users\x\AppData\Local\Voxweave\runtime\backend\.venv\Scripts\python.exe"));
        assert_eq!(l.data(), PathBuf::from(r"C:\Users\x\AppData\Local\Voxweave\data"));
        assert_eq!(l.backend_log(), PathBuf::from(r"C:\Users\x\AppData\Local\Voxweave\data\backend.log"));
        assert_eq!(l.record(), PathBuf::from(r"C:\Users\x\AppData\Local\Voxweave\runtime\setup.json"));
        assert_eq!(l.base_model(), PathBuf::from(r"C:\Users\x\AppData\Local\Voxweave\data\pretrained\vec-768-layer-12.onnx"));
        assert_eq!(l.backend_source(), PathBuf::from(r"C:\Program\Voxweave\backend"));
        assert_eq!(l.uv(), PathBuf::from(r"C:\Program\Voxweave\bin\uv.exe"));
        assert_eq!(l.bootstrap(), PathBuf::from(r"C:\Program\Voxweave\setup\bootstrap.py"));
    }

    #[test]
    fn the_setup_command_is_uv_running_the_bootstrap_with_python_3_12() {
        let (program, args) = layout().setup_command("0.1.0");
        assert_eq!(program, PathBuf::from(r"C:\Program\Voxweave\bin\uv.exe"));
        let args: Vec<String> = args.iter().map(|a| a.to_string_lossy().into_owned()).collect();
        assert_eq!(
            args,
            [
                "run", "--no-project", "--python", "3.12", r"C:\Program\Voxweave\setup\bootstrap.py",
                "--root", r"C:\Users\x\AppData\Local\Voxweave", "--resources", r"C:\Program\Voxweave",
                "--app-version", "0.1.0",
            ]
        );
    }

    fn value(env: &[(&'static str, OsString)], name: &str) -> String {
        env.iter().find(|(key, _)| *key == name).map(|(_, v)| v.to_string_lossy().into_owned()).unwrap_or_default()
    }

    #[test]
    fn uv_keeps_its_cache_and_python_under_the_root() {
        let env = layout().setup_env();
        assert_eq!(value(&env, "UV_CACHE_DIR"), r"C:\Users\x\AppData\Local\Voxweave\runtime\uv-cache");
        assert_eq!(value(&env, "UV_PYTHON_INSTALL_DIR"), r"C:\Users\x\AppData\Local\Voxweave\runtime\python");
        assert_eq!(value(&env, "UV_PYTHON_PREFERENCE"), "only-managed");
        assert_eq!(value(&env, "UV_PYTHON_INSTALL_REGISTRY"), "0");
        assert_eq!(value(&env, "UV_NO_CONFIG"), "1");
        assert_eq!(value(&env, "PYTHONNOUSERSITE"), "1");
        assert_eq!(value(&env, "PYTHONSAFEPATH"), "1");
    }

    #[test]
    fn the_backend_runs_from_the_bundled_source_with_its_data_under_the_root() {
        let env = layout().backend_env();
        assert_eq!(value(&env, "PYTHONPATH"), r"C:\Program\Voxweave\backend");
        assert_eq!(value(&env, "AIVOICE_DATA"), r"C:\Users\x\AppData\Local\Voxweave\data");
        assert_eq!(value(&env, "AIVOICE_INSTALLED"), "1");
        assert_eq!(value(&env, "PYTHONDONTWRITEBYTECODE"), "1", "nothing is written into the program folder");
        assert_eq!(value(&env, "PYTHONNOUSERSITE"), "1");
        assert_eq!(value(&env, "PYTHONSAFEPATH"), "1");
        for name in ["AIVOICE_DATA", "HF_HOME"] {
            assert!(Path::new(&value(&env, name)).starts_with(r"C:\Users\x\AppData\Local\Voxweave"), "{name}");
        }
    }

    #[test]
    fn the_users_own_uv_and_python_variables_are_dropped() {
        let names = ["Path", "UV_INDEX_URL", "uv_cache_dir", "VIRTUAL_ENV", "PythonPath", "PYTHONHOME", "UVICORN_X", "HOME"];
        let dropped = inherited_to_drop(names.iter().map(OsString::from));
        assert_eq!(dropped, ["UV_INDEX_URL", "uv_cache_dir", "VIRTUAL_ENV", "PythonPath", "PYTHONHOME"].map(OsString::from));
    }

    fn finished_setup(dir: &Path, record: &str) -> Installed {
        let l = Installed { root: dir.join("root"), resources: dir.join("program") };
        std::fs::create_dir_all(l.python().parent().unwrap()).unwrap();
        std::fs::write(l.python(), b"").unwrap();
        std::fs::create_dir_all(l.base_model().parent().unwrap()).unwrap();
        std::fs::write(l.base_model(), b"").unwrap();
        std::fs::write(l.record(), record).unwrap();
        l
    }

    #[test]
    fn setup_is_complete_only_for_the_same_version_with_everything_still_there() {
        let dir = scratch("complete");
        let l = finished_setup(&dir, r#"{"version": "0.1.0", "python": "3.12", "completed": "x"}"#);
        assert!(l.setup_complete("0.1.0"));
        assert!(!l.setup_complete("0.2.0"), "an update runs the setup again");
        std::fs::remove_file(l.base_model()).unwrap();
        assert!(!l.setup_complete("0.1.0"), "a missing base model runs the setup again");
        std::fs::write(l.base_model(), b"").unwrap();
        std::fs::remove_file(l.python()).unwrap();
        assert!(!l.setup_complete("0.1.0"), "a missing environment runs the setup again");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_missing_or_unreadable_record_means_setup_is_needed() {
        let dir = scratch("record");
        let l = finished_setup(&dir, "{not json");
        assert!(!l.setup_complete("0.1.0"));
        std::fs::write(l.record(), r#"{"python": "3.12"}"#).unwrap();
        assert!(!l.setup_complete("0.1.0"));
        std::fs::remove_file(l.record()).unwrap();
        assert!(!l.setup_complete("0.1.0"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn progress_and_final_lines_are_read() {
        assert_eq!(
            parse_line(r#"{"step": "packages", "percent": 37, "message": "Installing"}"#),
            Some(Line::Progress { step: "packages".into(), percent: 37, message: "Installing".into() })
        );
        assert_eq!(parse_line(r#"{"done": true, "ok": true, "error": null}"#), Some(Line::Done { ok: true, error: None }));
        assert_eq!(
            parse_line("{\"done\": true, \"ok\": false, \"error\": \"no network\"}\r\n"),
            Some(Line::Done { ok: false, error: Some("no network".into()) })
        );
        assert_eq!(
            parse_line(r#"{"step": "model", "percent": 250.5, "message": "x"}"#),
            Some(Line::Progress { step: "model".into(), percent: 100, message: "x".into() })
        );
    }

    #[test]
    fn other_output_is_not_a_line() {
        for text in ["", "Downloading cpython", "{}", "[1, 2]", r#"{"step": "x"}"#, r#"{"done": false}"#] {
            assert_eq!(parse_line(text), None, "{text}");
        }
    }

    #[test]
    fn the_state_follows_a_setup_from_start_to_ready() {
        let l = layout();
        assert_eq!(SetupState::dev().mode, "dev");
        assert_eq!(SetupState::dev().status, "ready");
        assert_eq!(SetupState::installed(&l, true).status, "ready");
        let mut state = SetupState::installed(&l, false);
        assert_eq!((state.mode, state.status, state.step.as_str(), state.percent), ("installed", "running", "python", 0));
        assert_eq!(state.root, r"C:\Users\x\AppData\Local\Voxweave");
        assert_eq!(state.data_dir, r"C:\Users\x\AppData\Local\Voxweave\data");
        state.progress("packages".into(), 40, "Installing".into());
        state.progress("packages".into(), 30, "Installing".into());
        assert_eq!(state.percent, 40, "the bar never goes backwards");
        state.fail("no network".into());
        assert_eq!((state.status, state.error.as_deref()), ("failed", Some("no network")));
        state.restart(false);
        assert_eq!((state.status, state.percent, state.error.as_deref()), ("running", 0, None));
        state.succeed();
        assert_eq!((state.status, state.percent), ("ready", 100));
    }

    #[test]
    fn the_state_is_sent_to_the_window_with_these_names() {
        let json = serde_json::to_value(SetupState::installed(&layout(), false)).unwrap();
        let mut names: Vec<&str> = json.as_object().unwrap().keys().map(String::as_str).collect();
        names.sort();
        assert_eq!(
            names,
            ["can_retry", "data_dir", "error", "first_time", "message", "mode", "percent", "root", "setup_log", "status", "step"]
        );
    }

    #[test]
    fn a_setup_is_the_first_one_until_a_record_of_a_finished_one_exists() {
        let dir = scratch("first-time");
        let l = Installed { root: dir.join("root"), resources: dir.join("program") };
        let state = SetupState::installed(&l, false);
        assert!(state.first_time && state.can_retry);
        std::fs::create_dir_all(l.runtime()).unwrap();
        std::fs::write(l.record(), r#"{"version": "0.0.9"}"#).unwrap();
        assert!(!SetupState::installed(&l, false).first_time, "an update only checks and downloads what changed");
        let mut state = state;
        state.restart(false);
        assert!(!state.first_time);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_broken_app_shows_its_reason_and_offers_no_retry() {
        let state = SetupState::broken(MISSING_FILE);
        assert_eq!((state.mode, state.status), ("installed", "failed"));
        assert_eq!(state.error.as_deref(), Some(MISSING_FILE));
        assert!(!state.can_retry && !state.first_time);
    }

    #[test]
    fn failures_without_a_message_from_the_bootstrap_are_explained() {
        let log = Path::new(r"C:\r\runtime\setup.log");
        assert_eq!(failure_message(Some("no space".into()), true, Some(1), &[], log), "no space");
        let not_started = failure_message(None, false, Some(2), &["error: failed to download".into()], log);
        assert!(not_started.contains("Python could not be downloaded") && not_started.contains("internet"));
        assert!(not_started.contains("failed to download"));
        let crashed = failure_message(Some("  ".into()), true, Some(3), &[], log);
        assert!(crashed.contains("exit code 3") && crashed.contains(r"C:\r\runtime\setup.log"));
    }
}
