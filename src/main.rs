mod app;
mod audio;
mod cli;
mod config;
mod device;
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
use clap::Parser;
use crossterm::{
    event::{self, Event},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{backend::CrosstermBackend, Terminal};

use app::App;
use cli::{Cli, Commands};
use cli::record::RecordOptions;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use web::state::{ActionRx, BrowserState, StateTx, WebAction};

fn main() -> Result<()> {
    // Load .env from the current working directory.  Silently ignored if
    // absent; shell env vars take precedence.
    dotenvy::dotenv().ok();

    let cli = Cli::parse();

    match cli.command.unwrap_or_else(|| Commands::Record(cli::record::RecordArgs::default())) {
        Commands::Config(args)   => cli::config::run(&args),
        Commands::Export(args)   => cli::export::run(&args),
        Commands::Playback(args) => cli::playback::run(&args),
        Commands::Record(args)   => {
            let opts = RecordOptions::resolve(&args)?;
            run_record(opts)
        }
    }
}

fn run_record(opts: RecordOptions) -> Result<()> {
    let (web_port, state_tx, action_rx) = web::spawn(opts.browser_port)?;

    if opts.browser_open {
        let _ = std::process::Command::new("xdg-open")
            .arg(format!("http://localhost:{web_port}/"))
            .stderr(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .spawn();
    }

    if opts.web_mode {
        return run_headless(web_port, state_tx, action_rx, &opts);
    }

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;

    let result = run_tui(state_tx, action_rx, &opts);

    // Always restore terminal, even on error.
    let _ = disable_raw_mode();
    let _ = execute!(io::stdout(), LeaveAlternateScreen);

    result
}

// ── Drain browser actions ─────────────────────────────────────────────────────

/// Drain all pending browser actions.  Returns `true` if any action triggered
/// a quit (so the caller can break its event loop).
fn drain_actions(app: &mut App, action_rx: &ActionRx) -> bool {
    while let Ok(action) = action_rx.try_recv() {
        match action {
            WebAction::Key(name) => {
                // Map action names to the same key codes the TUI uses, then
                // feed them through the existing state machine via handle_key.
                let code = match name.as_str() {
                    "start" | "stop"   => KeyCode::Char(' '),
                    "punch"            => KeyCode::Char('p'),
                    "play"  | "pause"  => KeyCode::Char('l'),
                    "jump_back"        => KeyCode::Char('j'),
                    "jump_forward"     => KeyCode::Char('k'),
                    "next_chapter"     => KeyCode::Char('n'),
                    "retry_mic"        => KeyCode::Char('r'),
                    "edit_session"     => KeyCode::Char('e'),
                    "quit"             => KeyCode::Char('q'),
                    _                  => continue,
                };
                let key = KeyEvent::new(code, KeyModifiers::NONE);
                if app.handle_key(key) {
                    return true;
                }
            }
        }
    }
    false
}

// ── Headless loop ─────────────────────────────────────────────────────────────

fn run_headless(port: u16, state_tx: StateTx, action_rx: ActionRx, opts: &RecordOptions) -> Result<()> {
    eprintln!("rustcorder [web_mode]: open http://localhost:{port}/ in your browser");
    eprintln!("rustcorder [web_mode]: press Ctrl-C to quit");

    let sigterm = Arc::new(AtomicBool::new(false));
    signal_hook::flag::register(signal_hook::consts::SIGTERM, Arc::clone(&sigterm))?;

    let mut app = App::new_with_options(
        opts.punch_back_time,
        opts.discard_short_clips,
        opts.discard_duration_secs,
    );
    app.detect_mic();

    loop {
        if sigterm.load(Ordering::Relaxed) {
            app.emergency_stop();
        }
        if app.should_quit { break; }

        if drain_actions(&mut app, &action_rx) { break; }
        app.tick();
        let _ = state_tx.send(BrowserState::from_app(&app));

        if app.should_quit { break; }
        std::thread::sleep(Duration::from_millis(40));
    }

    Ok(())
}

// ── TUI loop ──────────────────────────────────────────────────────────────────

fn run_tui(state_tx: StateTx, action_rx: ActionRx, opts: &RecordOptions) -> Result<()> {
    let sigterm = Arc::new(AtomicBool::new(false));
    signal_hook::flag::register(signal_hook::consts::SIGTERM, Arc::clone(&sigterm))?;

    let backend = CrosstermBackend::new(io::stdout());
    let mut terminal = Terminal::new(backend)?;

    let mut app = App::new_with_options(
        opts.punch_back_time,
        opts.discard_short_clips,
        opts.discard_duration_secs,
    );
    app.detect_mic();

    loop {
        // ── Drain browser actions ─────────────────────────────────────────
        if drain_actions(&mut app, &action_rx) { break; }

        // ── Render ────────────────────────────────────────────────────────
        terminal.draw(|f| render::draw(f, &app))?;

        // ── Broadcast state to browser clients ────────────────────────────
        let _ = state_tx.send(BrowserState::from_app(&app));

        // ── Process OS signals ────────────────────────────────────────────
        if sigterm.load(Ordering::Relaxed) {
            app.emergency_stop();
        }
        if app.should_quit { break; }

        // ── Handle input events (non-blocking, 40 ms poll) ────────────────
        if event::poll(Duration::from_millis(40))? {
            if let Event::Key(key) = event::read()? {
                if app.handle_key(key) { break; }
            }
        }

        // ── Tick ──────────────────────────────────────────────────────────
        app.tick();
        if app.should_quit { break; }
    }

    Ok(())
}
