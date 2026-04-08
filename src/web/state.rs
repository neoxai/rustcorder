//! Shared state snapshot sent to browser clients, and action back-channel.

use serde::Serialize;

use crate::app::{App, AppMode};

// ── BrowserState ──────────────────────────────────────────────────────────────

/// Minimal, cheaply-cloneable snapshot of `App` broadcast to all connected
/// browser clients via WebSocket and served at `GET /state`.
#[derive(Clone, Debug, Serialize)]
pub struct BrowserState {
    /// Current recorder mode, e.g. `"Recording"`, `"Ready"`.
    pub mode: String,
    pub book: String,
    pub chapter: u32,
    pub part: u32,
    /// Seconds elapsed since the current recording started; 0 when not recording.
    pub elapsed_secs: f64,
    /// Most recent RMS level in dBFS.
    pub rms_dbfs: f32,
    /// True when the silence watchdog has fired (>15 s of silence while recording).
    pub silence_warning: bool,
    /// Current EPUB position as a CFI string, or `null` when no EPUB is loaded.
    pub epub_cfi: Option<String>,
    /// Absolute filesystem path to the `.epub` file served by `GET /epub`.
    /// `null` when no EPUB is loaded.  Not shown in the browser UI; used
    /// internally by the `/epub` endpoint.
    pub epub_path: Option<String>,
    /// Absolute timeline position in seconds for the current chapter.
    pub timeline_pos_secs: f64,
    /// Key-hint string matching the TUI footer for the current mode.
    pub footer_hints: String,
}

impl Default for BrowserState {
    fn default() -> Self {
        BrowserState {
            mode: "Init".to_string(),
            book: String::new(),
            chapter: 1,
            part: 1,
            elapsed_secs: 0.0,
            rms_dbfs: -100.0,
            silence_warning: false,
            epub_cfi: None,
            epub_path: None,
            timeline_pos_secs: 0.0,
            footer_hints: String::new(),
        }
    }
}

impl BrowserState {
    /// Build a snapshot from the current `App`.  Called on every tick.
    pub fn from_app(app: &App) -> Self {
        let mode = match app.mode {
            AppMode::Setup => "Setup",
            AppMode::Ready => "Ready",
            AppMode::PreCheck => "PreCheck",
            AppMode::Recording => "Recording",
            AppMode::PostRecording => "PostRecording",
            AppMode::PunchRollback => "PunchRollback",
            AppMode::MicError => "MicError",
            AppMode::Fatal => "Fatal",
        }
        .to_string();

        let elapsed_secs = app
            .record_start
            .map(|s| s.elapsed().as_secs_f64())
            .unwrap_or(0.0);

        let epub_cfi = app.epub.as_ref().map(|e| e.cfi.to_cfi_string());

        // Build the epub filesystem path from session dir + epub filename.
        let epub_path = app.epub.as_ref().map(|e| {
            app.session
                .output_dir()
                .join(e.epub_filename())
                .to_string_lossy()
                .into_owned()
        });

        let mode_keys = match app.mode {
            AppMode::Setup => "[Enter] Confirm   [Tab] Switch field   [Esc] Cancel",
            AppMode::Ready => "[Space] Start   [E] Edit session   [R] Re-detect mic   [Q] Quit",
            AppMode::PreCheck => "[Esc] Abort check",
            AppMode::Recording => "[Space] Stop   [P] Punch and roll",
            AppMode::PostRecording => "[Y/Enter] Continue chapter   [N] Chapter complete",
            AppMode::PunchRollback => "[Space] Punch in here   [Esc] Cancel",
            AppMode::MicError => "[R] Retry detection   [Q] Quit",
            AppMode::Fatal => "[Q] Quit",
        };
        let footer_hints = if app.epub.is_some() {
            format!("{}   [← →] Scroll text", mode_keys)
        } else {
            mode_keys.to_string()
        };

        BrowserState {
            mode,
            book: app.session.book.clone(),
            chapter: app.session.chapter,
            part: app.session.part,
            elapsed_secs,
            rms_dbfs: app.last_rms_db,
            silence_warning: app.silence_warning,
            epub_cfi,
            epub_path,
            timeline_pos_secs: app.session.timeline_pos,
            footer_hints,
        }
    }
}

// ── Action back-channel ───────────────────────────────────────────────────────

/// Actions the browser can send back to the main app loop.
pub enum WebAction {
    /// User scrolled the epub.js rendition; update the Rust position.
    EpubSeek(crate::epub::EpubCfi),
}

/// Sender half — held by each WebSocket connection handler.
pub type ActionTx = std::sync::mpsc::SyncSender<WebAction>;
/// Receiver half — held by the main app loop and drained each tick.
pub type ActionRx = std::sync::mpsc::Receiver<WebAction>;

// ── Convenience alias ─────────────────────────────────────────────────────────

/// The sender half of the state watch channel; passed into the app event loop.
pub type StateTx = tokio::sync::watch::Sender<BrowserState>;
