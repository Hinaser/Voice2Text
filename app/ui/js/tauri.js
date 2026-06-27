// Thin re-exports of the Tauri globals so the rest of the UI imports from one
// place instead of reaching into window.__TAURI__ everywhere.
export const { listen } = window.__TAURI__.event;
export const { invoke } = window.__TAURI__.core;
export const appWindow = window.__TAURI__.window.getCurrentWindow();
