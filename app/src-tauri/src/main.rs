// Console stays visible for now (shows model-load + pipeline logs during early
// builds). Add `#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]`
// later to hide it in the shipped release.

mod commands;
mod config;
mod paths;
mod pipeline;
mod streaming;
mod summary;

use std::path::PathBuf;
use std::str::FromStr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use tauri::Manager;
use tauri_plugin_global_shortcut::{GlobalShortcutExt, Shortcut, ShortcutState};

use commands::AppState;
use config::Config;

/// Show the overlay if hidden, hide it if visible.
fn toggle_overlay(app: &tauri::AppHandle) {
    if let Some(win) = app.get_webview_window("overlay") {
        if matches!(win.is_visible(), Ok(true)) {
            let _ = win.hide();
        } else {
            let _ = win.show();
            let _ = win.set_focus();
        }
    }
}

/// Register the configured global show/hide hotkey, if any.
fn register_hotkey(app: &tauri::AppHandle, hotkey: &str) {
    let hk = hotkey.trim();
    if hk.is_empty() {
        return;
    }
    match Shortcut::from_str(hk) {
        Ok(shortcut) => {
            if let Err(e) = app.global_shortcut().register(shortcut) {
                eprintln!("hotkey '{hk}' could not be registered: {e}");
            }
        }
        Err(e) => eprintln!("invalid hotkey '{hk}': {e}"),
    }
}

fn main() {
    let running = Arc::new(AtomicBool::new(true));

    tauri::Builder::default()
        .plugin(
            tauri_plugin_global_shortcut::Builder::new()
                .with_handler(|app, _shortcut, event| {
                    if event.state == ShortcutState::Pressed {
                        toggle_overlay(app);
                    }
                })
                .build(),
        )
        .setup({
            let running = running.clone();
            move |app| {
                let config_path = app
                    .path()
                    .app_config_dir()
                    .unwrap_or_else(|_| PathBuf::from("."))
                    .join("config.json");
                let config = Arc::new(Mutex::new(Config::load(&config_path)));
                // Materialize the file on first run so it's discoverable/editable.
                let _ = config.lock().unwrap().save(&config_path);

                register_hotkey(&app.handle().clone(), &config.lock().unwrap().hotkey.clone());

                app.manage(AppState { config: config.clone(), config_path });

                let handle = app.handle().clone();
                let run_flag = running.clone();
                std::thread::spawn(move || {
                    if let Err(e) = pipeline::run(handle, run_flag, config) {
                        eprintln!("pipeline error: {e}");
                    }
                });
                Ok(())
            }
        })
        .on_window_event({
            let running = running.clone();
            move |_window, event| {
                if let tauri::WindowEvent::Destroyed = event {
                    running.store(false, Ordering::SeqCst);
                }
            }
        })
        .invoke_handler(tauri::generate_handler![
            commands::get_config,
            commands::set_config,
            commands::save_dir,
            commands::open_save_dir,
            commands::export_transcript,
            commands::summarize,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
