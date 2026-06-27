//! Console renderer: a scrolling transcript of finalized lines (word-wrapped to
//! the terminal width) with an in-place live partial pinned below it.
//!
//! Model: finalized lines are printed permanently and scroll up like a normal
//! transcript; the still-changing partial lives on the line(s) just below and is
//! redrawn in place using *relative* cursor moves (so it survives scrolling).
//! Both finals and the partial are word-wrapped at the right edge.

use std::io::Write;

use windows_sys::Win32::System::Console::{
    GetConsoleMode, GetConsoleScreenBufferInfo, GetStdHandle, SetConsoleMode,
    CONSOLE_SCREEN_BUFFER_INFO, ENABLE_VIRTUAL_TERMINAL_PROCESSING, STD_OUTPUT_HANDLE,
};

pub struct Live {
    width: usize,
    /// Physical rows the current partial occupies (cursor rests at its start).
    partial_rows: usize,
}

impl Live {
    pub fn new() -> Self {
        let width = enable_vt_and_get_width().unwrap_or(100).clamp(20, 400);
        println!("Voice2Text — live captions (streaming). Speakers are auto-labeled. Ctrl+C to stop.\n");
        std::io::stdout().flush().ok();
        Self { width, partial_rows: 0 }
    }

    /// Redraw the live (not-yet-final) partial in place.
    pub fn set_partial(&mut self, text: &str) {
        let lines = wrap(text, self.width);
        let mut out = String::new();
        out.push_str("\x1b[0J"); // erase old partial (cursor is at its start)
        out.push_str(&lines.join("\r\n"));
        // return cursor to the partial's start line, column 0
        if lines.len() > 1 {
            out.push_str(&format!("\x1b[{}A", lines.len() - 1));
        }
        out.push('\r');
        print!("{out}");
        std::io::stdout().flush().ok();
        self.partial_rows = lines.len();
    }

    /// Commit a finalized line: erase the partial, print the line permanently
    /// (wrapped, scrolls), and leave the cursor at the new partial start.
    pub fn commit(&mut self, text: &str) {
        let mut out = String::new();
        out.push_str("\x1b[0J"); // erase partial
        for line in wrap(text, self.width) {
            out.push_str(&line);
            out.push_str("\r\n");
        }
        print!("{out}");
        std::io::stdout().flush().ok();
        self.partial_rows = 0;
    }
}

/// Greedy word-wrap to `width` columns; hard-splits any word longer than width.
fn wrap(text: &str, width: usize) -> Vec<String> {
    let mut lines: Vec<String> = Vec::new();
    let mut cur = String::new();
    for word in text.split_whitespace() {
        if cur.is_empty() {
            cur = word.to_string();
        } else if cur.chars().count() + 1 + word.chars().count() <= width {
            cur.push(' ');
            cur.push_str(word);
        } else {
            lines.push(std::mem::take(&mut cur));
            cur = word.to_string();
        }
        while cur.chars().count() > width {
            let head: String = cur.chars().take(width).collect();
            lines.push(head);
            cur = cur.chars().skip(width).collect();
        }
    }
    if !cur.is_empty() {
        lines.push(cur);
    }
    if lines.is_empty() {
        lines.push(String::new());
    }
    lines
}

fn enable_vt_and_get_width() -> Option<usize> {
    unsafe {
        let h = GetStdHandle(STD_OUTPUT_HANDLE);
        let mut mode = 0u32;
        if GetConsoleMode(h, &mut mode) != 0 {
            SetConsoleMode(h, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
        }
        let mut info: CONSOLE_SCREEN_BUFFER_INFO = std::mem::zeroed();
        if GetConsoleScreenBufferInfo(h, &mut info) != 0 {
            return Some((info.srWindow.Right - info.srWindow.Left + 1) as usize);
        }
    }
    None
}
