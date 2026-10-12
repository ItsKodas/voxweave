use std::net::TcpListener;
use std::path::PathBuf;
use std::process::{Child, Command};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

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
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    cmd.spawn()
}

/// Keeps the backend alive. A restarted backend has no live session, so live mode comes back stopped.
fn supervise(port: u16, child: Arc<Mutex<Option<Child>>>, quitting: Arc<AtomicBool>) {
    thread::spawn(move || {
        while !quitting.load(Ordering::SeqCst) {
            match spawn_backend(port) {
                Ok(process) => *child.lock().unwrap() = Some(process),
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
                    *guard = None;
                    break;
                }
            }
            thread::sleep(Duration::from_secs(1));
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
    supervise(port, child.clone(), quitting.clone());

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
        .setup(|app| {
            app.global_shortcut()
                .register(Shortcut::new(Some(Modifiers::CONTROL | Modifiers::ALT), Code::KeyV))?;

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
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building tauri application");

    app.run(move |_handle, event| {
        if let RunEvent::Exit = event {
            quitting.store(true, Ordering::SeqCst);
            if let Some(mut process) = child.lock().unwrap().take() {
                let _ = process.kill();
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
    fn backend_dir_honours_env() {
        // SAFETY: only this test touches this variable.
        unsafe { std::env::set_var("AIVOICE_BACKEND_DIR", "C:/x/backend") };
        assert_eq!(backend_dir(), PathBuf::from("C:/x/backend"));
        unsafe { std::env::remove_var("AIVOICE_BACKEND_DIR") };
    }
}
