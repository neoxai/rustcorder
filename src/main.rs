mod app;
mod audio;
mod config;
mod device;
mod epub;
mod export;
mod playback;
mod render;
mod session;
mod wav;
mod web;

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
use web::state::{BrowserState, StateTx};

fn main() -> Result<()> {
    // Load .env from the current working directory into the process environment.
    // Silently ignored if the file is absent; shell env vars take precedence.
    dotenvy::dotenv().ok();

    // `--config` runs before the TUI — plain stdin/stdout, no raw mode.
    if std::env::args().any(|a| a == "--config") {
        return config::run_config();
    }

    // `--export` reconstructs a chapter from its timeline and clip files.
    if std::env::args().any(|a| a == "--export") {
        let args: Vec<String> = std::env::args().collect();
        let book = args
            .windows(2)
            .find(|w| w[0] == "--book")
            .map(|w| w[1].clone())
            .ok_or_else(|| anyhow::anyhow!("--book <name> is required with --export"))?;
        let chapter: u32 = args
            .windows(2)
            .find(|w| w[0] == "--chapter")
            .and_then(|w| w[1].parse().ok())
            .ok_or_else(|| anyhow::anyhow!("--chapter <n> is required with --export"))?;
        let crossfade_ms = std::env::var("CROSSFADE_TIME")
            .ok()
            .and_then(|v| v.trim().trim_end_matches("ms").parse::<f64>().ok())
            .unwrap_or(10.0);
        let book_dir = std::path::Path::new(&book);
        let out = export::timeline::export_chapter(book_dir, chapter, crossfade_ms)?;
        println!("Exported: {}", out.display());
        return Ok(());
    }

    // `--playback` runs before the TUI — terminal only, no alternate screen.
    if std::env::args().any(|a| a == "--playback") {
        let args: Vec<String> = std::env::args().collect();
        let player_type = args
            .windows(2)
            .find(|w| w[0] == "--type")
            .and_then(|w| w[1].parse::<u8>().ok())
            .unwrap_or(1);
        let file = args
            .windows(2)
            .find(|w| w[0] == "--file")
            .map(|w| std::path::PathBuf::from(&w[1]))
            .ok_or_else(|| anyhow::anyhow!("--file <path> is required with --playback"))?;
        return playback::run(player_type, &file);
    }

    // Start the local web server.  Returns the bound port and a channel sender
    // for pushing state snapshots to connected clients.
    let (web_port, state_tx) = web::spawn()?;

    let web_mode = std::env::var("WEB_MODE")
        .map(|v| v.trim().eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    if web_mode {
        return run_headless(web_port, state_tx);
    }

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;

    let result = run(state_tx);

    // Always restore terminal, even on error.
    let _ = disable_raw_mode();
    let _ = execute!(io::stdout(), LeaveAlternateScreen);

    result
}

fn run_headless(port: u16, state_tx: StateTx) -> Result<()> {
    eprintln!("rustcorder [WEB_MODE]: open http://localhost:{port}/ in your browser");
    eprintln!("rustcorder [WEB_MODE]: press Ctrl-C to quit");

    let sigterm = Arc::new(AtomicBool::new(false));
    signal_hook::flag::register(signal_hook::consts::SIGTERM, Arc::clone(&sigterm))?;

    let mut app = App::new();
    app.detect_mic();

    loop {
        if sigterm.load(Ordering::Relaxed) {
            app.emergency_stop();
        }
        if app.should_quit {
            break;
        }
        app.tick();
        let _ = state_tx.send(BrowserState::from_app(&app));
        if app.should_quit {
            break;
        }
        std::thread::sleep(Duration::from_millis(40));
    }

    Ok(())
}

fn run(state_tx: StateTx) -> Result<()> {
    // Register a SIGTERM handler so the app can shut down cleanly when asked
    // to exit by the OS / process manager.
    let sigterm = Arc::new(AtomicBool::new(false));
    signal_hook::flag::register(signal_hook::consts::SIGTERM, Arc::clone(&sigterm))?;

    let backend = CrosstermBackend::new(io::stdout());
    let mut terminal = Terminal::new(backend)?;

    let mut app = App::new();

    // Attempt initial mic detection (non-fatal; UI will show the error).
    app.detect_mic();

    loop {
        // ── Render ────────────────────────────────────────────────────────
        // Inform the app of the current terminal width so the EPUB pane can
        // re-wrap if the terminal was resized since the last frame.
        let epub_pane_width = terminal.size().map(|s| s.width).unwrap_or(80);
        app.epub_set_pane_width(epub_pane_width);
        terminal.draw(|f| render::draw(f, &app))?;

        // ── Broadcast state to browser clients ────────────────────────────
        let _ = state_tx.send(BrowserState::from_app(&app));

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
