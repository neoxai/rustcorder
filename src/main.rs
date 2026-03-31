mod app;
mod audio;
mod device;
mod render;
mod session;
mod wav;

use std::io;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use crossterm::{
    event::{self, Event},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{backend::CrosstermBackend, Terminal};

use app::App;

fn main() -> Result<()> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;

    let result = run();

    // Always restore terminal, even on error.
    let _ = disable_raw_mode();
    let _ = execute!(io::stdout(), LeaveAlternateScreen);

    result
}

fn run() -> Result<()> {
    let unsafe_mode = std::env::args().any(|a| a == "--unsafe");

    // Register a SIGTERM handler so the app can shut down cleanly when asked
    // to exit by the OS / process manager.
    let sigterm = Arc::new(AtomicBool::new(false));
    signal_hook::flag::register(signal_hook::consts::SIGTERM, Arc::clone(&sigterm))?;

    let backend = CrosstermBackend::new(io::stdout());
    let mut terminal = Terminal::new(backend)?;

    let mut app = App::new(unsafe_mode);

    // Attempt initial mic detection (non-fatal; UI will show the error).
    app.detect_mic();

    loop {
        // ── Render ────────────────────────────────────────────────────────
        terminal.draw(|f| render::draw(f, &app))?;

        // ── Process OS signals ────────────────────────────────────────────
        if sigterm.load(Ordering::Relaxed) {
            app.emergency_stop();
        }

        if app.should_quit {
            break;
        }

        // ── Handle input events (non-blocking, 40 ms poll) ────────────────
        if event::poll(Duration::from_millis(40))? {
            if let Event::Key(key) = event::read()? {
                if app.handle_key(key) {
                    break;
                }
            }
        }

        // ── Tick: drain audio channel, advance state ───────────────────────
        app.tick();

        if app.should_quit {
            break;
        }
    }

    Ok(())
}
