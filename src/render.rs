use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, Paragraph, Wrap},
    Frame,
};

use crate::app::{App, AppMode, SetupField};
use crate::audio::{SILENCE_THRESHOLD_DB, SILENCE_WARNING_SECONDS};

// ── Colour palette ────────────────────────────────────────────────────────────

const RED: Color = Color::Red;
const GREEN: Color = Color::Green;
const YELLOW: Color = Color::Yellow;
const CYAN: Color = Color::Cyan;
const WHITE: Color = Color::White;
const DARK_GRAY: Color = Color::DarkGray;

fn bold(c: Color) -> Style {
    Style::default().fg(c).add_modifier(Modifier::BOLD)
}
fn dim() -> Style {
    Style::default().fg(DARK_GRAY)
}

// ── Entry point ───────────────────────────────────────────────────────────────

pub fn draw(f: &mut Frame, app: &App) {
    let area = f.area();

    // Outer chrome
    let outer = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(CYAN))
        .title(Span::styled(
            " RUSTCORDER ",
            bold(CYAN).add_modifier(Modifier::BOLD),
        ));
    f.render_widget(outer, area);

    // Inner usable area (inside the border)
    let inner = inner_rect(area);

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // header bar
            Constraint::Min(0),    // body
            Constraint::Length(1), // keybind footer
        ])
        .split(inner);

    draw_header(f, app, chunks[0]);
    draw_body(f, app, chunks[1]);
    draw_footer(f, app, chunks[2]);
}

// ── Header ────────────────────────────────────────────────────────────────────

fn draw_header(f: &mut Frame, app: &App, area: Rect) {
    let state_label = match app.mode {
        AppMode::Setup => Span::styled("[ SETUP ]", bold(YELLOW)),
        AppMode::Ready => Span::styled("[ READY ]", bold(GREEN)),
        AppMode::PreCheck => Span::styled("[ CHECKING ]", bold(YELLOW)),
        AppMode::Recording => Span::styled("[ *** RECORDING *** ]", bold(RED)),
        AppMode::PostRecording => Span::styled("[ STOPPED ]", bold(YELLOW)),
        AppMode::PunchRollback => Span::styled("[ ROLLING BACK ]", bold(YELLOW)),
        AppMode::MicError => Span::styled("[ MIC ERROR ]", bold(RED)),
        AppMode::Fatal => Span::styled("[ FATAL ERROR ]", bold(RED)),
    };

    let book_label = if app.session.book.is_empty() {
        Span::styled("(no book)", dim())
    } else {
        Span::styled(app.session.book.clone(), bold(WHITE))
    };

    let chapter_label = Span::styled(
        format!("Ch {:02}", app.session.chapter),
        Style::default().fg(WHITE),
    );

    let part_label = Span::styled(
        format!("Part {:03}", app.session.part),
        Style::default().fg(WHITE),
    );

    let sep = Span::styled("  │  ", dim());

    let line = Line::from(vec![
        book_label,
        sep.clone(),
        chapter_label,
        sep.clone(),
        part_label,
        Span::raw("  "),
        state_label,
    ]);

    let block = Block::default()
        .borders(Borders::BOTTOM)
        .border_style(dim());
    let paragraph = Paragraph::new(line).block(block).alignment(Alignment::Left);
    f.render_widget(paragraph, area);
}


// ── Body ──────────────────────────────────────────────────────────────────────

fn draw_body(f: &mut Frame, app: &App, area: Rect) {
    draw_body_mode(f, app, area);
}

fn draw_body_mode(f: &mut Frame, app: &App, area: Rect) {
    match app.mode {
        AppMode::Setup => draw_setup(f, app, area),
        AppMode::Ready => draw_ready(f, app, area),
        AppMode::PreCheck => draw_precheck(f, app, area),
        AppMode::Recording => draw_recording(f, app, area),
        AppMode::PostRecording => draw_post_recording(f, app, area),
        AppMode::PunchRollback => draw_punch_rollback(f, app, area),
        AppMode::MicError => draw_mic_error(f, app, area),
        AppMode::Fatal => draw_fatal(f, app, area),
    }
}

// ── Setup ─────────────────────────────────────────────────────────────────────

fn draw_setup(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Length(3), // book field
            Constraint::Length(3), // chapter field
            Constraint::Length(2), // error
            Constraint::Min(0),
        ])
        .split(area);

    // Title
    f.render_widget(
        Paragraph::new("Session Setup").style(bold(CYAN)),
        chunks[0],
    );

    // Book name field
    let book_style = if app.setup_field == SetupField::Book {
        bold(WHITE)
    } else {
        Style::default().fg(DARK_GRAY)
    };
    let book_cursor = if app.setup_field == SetupField::Book { "█" } else { "" };
    let book_block = Block::default()
        .borders(Borders::ALL)
        .border_style(if app.setup_field == SetupField::Book {
            bold(CYAN)
        } else {
            dim()
        })
        .title("Book name (used as directory)");
    f.render_widget(
        Paragraph::new(format!("{}{}", app.setup_book, book_cursor))
            .style(book_style)
            .block(book_block),
        chunks[1],
    );

    // Chapter field
    let ch_style = if app.setup_field == SetupField::Chapter {
        bold(WHITE)
    } else {
        Style::default().fg(DARK_GRAY)
    };
    let ch_cursor = if app.setup_field == SetupField::Chapter { "█" } else { "" };
    let ch_block = Block::default()
        .borders(Borders::ALL)
        .border_style(if app.setup_field == SetupField::Chapter {
            bold(CYAN)
        } else {
            dim()
        })
        .title("Starting chapter number");
    f.render_widget(
        Paragraph::new(format!("{}{}", app.setup_chapter, ch_cursor))
            .style(ch_style)
            .block(ch_block),
        chunks[2],
    );

    // Error
    if let Some(ref err) = app.setup_error {
        f.render_widget(
            Paragraph::new(err.as_str()).style(bold(RED)),
            chunks[3],
        );
    }
}

// ── Ready ─────────────────────────────────────────────────────────────────────

fn draw_ready(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(2), // mic info
            Constraint::Length(1), // blank
            Constraint::Length(1), // next file
            Constraint::Length(1), // timeline pos
            Constraint::Length(1), // blank
            Constraint::Length(2), // status msg
            Constraint::Min(0),
        ])
        .split(area);

    // Mic status
    let mic_line = match &app.mic {
        Some(m) => Line::from(vec![
            Span::styled("Microphone: ", dim()),
            Span::styled(m.description.clone(), bold(GREEN)),
            Span::styled("  ✓", bold(GREEN)),
        ]),
        None => Line::from(vec![Span::styled(
            "Microphone: NOT DETECTED",
            bold(RED),
        )]),
    };
    f.render_widget(Paragraph::new(mic_line), chunks[0]);

    // Next output file
    let file_line = Line::from(vec![
        Span::styled("Next file:  ", dim()),
        Span::styled(
            app.session.current_path().display().to_string(),
            Style::default().fg(CYAN),
        ),
    ]);
    f.render_widget(Paragraph::new(file_line), chunks[2]);

    // Timeline position
    let tl_line = Line::from(vec![
        Span::styled("Timeline:   ", dim()),
        Span::styled(
            format!("{:.3} s", app.session.timeline_pos),
            Style::default().fg(DARK_GRAY),
        ),
    ]);
    f.render_widget(Paragraph::new(tl_line), chunks[3]);

    // Status/warning message
    if let Some(ref msg) = app.status_msg {
        f.render_widget(
            Paragraph::new(msg.as_str())
                .style(bold(YELLOW))
                .wrap(Wrap { trim: true }),
            chunks[5],
        );
    }
}

// ── Pre-check ─────────────────────────────────────────────────────────────────

fn draw_precheck(f: &mut Frame, app: &App, area: Rect) {
    use crate::audio::SAMPLE_RATE;

    let pct = (app.precheck_frames_seen as f64
        / (crate::audio::PRECHECK_SECONDS * SAMPLE_RATE as f64))
        .min(1.0);

    let bar_width = area.width.saturating_sub(4) as usize;
    let filled = (pct * bar_width as f64) as usize;
    let bar: String = std::iter::repeat('█')
        .take(filled)
        .chain(std::iter::repeat('░').take(bar_width.saturating_sub(filled)))
        .collect();

    let signal_text = if app.precheck_signal_found {
        Span::styled("  ✓ Signal detected", bold(GREEN))
    } else {
        Span::styled("  — Listening for signal…", dim())
    };

    let spinner = SPINNER[app.activity_tick as usize % SPINNER.len()];
    let spinner_span = if app.precheck_signal_found {
        Span::styled(format!(" {} ", spinner), bold(GREEN))
    } else {
        Span::styled("   ", Style::default())
    };

    let lines = vec![
        Line::from(Span::styled(
            "Checking for signal (2 seconds)…",
            bold(YELLOW),
        )),
        Line::from(Span::raw("")),
        Line::from(vec![spinner_span, Span::raw(format!("[{}]", bar))]),
        Line::from(signal_text),
        Line::from(Span::raw("")),
        Line::from(Span::styled(
            "Check your microphone is unmuted and pointed at a sound source.",
            dim(),
        )),
    ];

    f.render_widget(Paragraph::new(lines), area);
}

// ── Recording ─────────────────────────────────────────────────────────────────

fn draw_recording(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // BIG "RECORDING" banner
            Constraint::Length(1), // blank
            Constraint::Length(1), // activity / VU
            Constraint::Length(1), // RMS value + silence warning
            Constraint::Length(1), // blank
            Constraint::Length(1), // elapsed time
            Constraint::Length(1), // output file
            Constraint::Min(0),
        ])
        .split(area);

    // Banner
    let banner = Paragraph::new(
        Line::from(Span::styled(
            "  ●  RECORDING  ●  ",
            Style::default()
                .fg(Color::Black)
                .bg(RED)
                .add_modifier(Modifier::BOLD),
        )),
    )
    .alignment(Alignment::Center);
    f.render_widget(banner, chunks[0]);

    // Activity / VU meter
    let vu = build_vu_meter(app.last_rms_db, area.width.saturating_sub(16) as usize);
    let spinner = if app.last_rms_db > SILENCE_THRESHOLD_DB {
        Span::styled(
            format!(" {} ", SPINNER[app.activity_tick as usize % SPINNER.len()]),
            bold(GREEN),
        )
    } else {
        Span::styled(" — ", dim())
    };
    f.render_widget(
        Paragraph::new(Line::from(vec![spinner, Span::raw(vu)])),
        chunks[2],
    );

    // RMS + silence warning
    let rms_line = if app.silence_warning {
        Line::from(Span::styled(
            format!(
                "  ⚠  NO AUDIO DETECTED FOR {:.0}s — CHECK MUTE BUTTON  ⚠",
                SILENCE_WARNING_SECONDS
            ),
            Style::default()
                .fg(Color::Black)
                .bg(YELLOW)
                .add_modifier(Modifier::BOLD | Modifier::SLOW_BLINK),
        ))
    } else {
        Line::from(vec![
            Span::styled("    Level: ", dim()),
            Span::styled(
                format!("{:.1} dBFS", app.last_rms_db),
                Style::default().fg(level_color(app.last_rms_db)),
            ),
        ])
    };
    f.render_widget(Paragraph::new(rms_line), chunks[3]);

    // Elapsed time
    if let Some(elapsed) = app.elapsed_recording() {
        let secs = elapsed.as_secs();
        let h = secs / 3600;
        let m = (secs % 3600) / 60;
        let s = secs % 60;
        f.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled("  Elapsed: ", dim()),
                Span::styled(
                    format!("{:02}:{:02}:{:02}", h, m, s),
                    bold(WHITE),
                ),
            ])),
            chunks[5],
        );
    }

    // Current file path
    if let Some(ref path) = app.active_file {
        f.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled("  File:    ", dim()),
                Span::styled(path.display().to_string(), Style::default().fg(CYAN)),
            ])),
            chunks[6],
        );
    }
}

// ── Post-recording ────────────────────────────────────────────────────────────

fn draw_post_recording(f: &mut Frame, app: &App, area: Rect) {
    let file_msg = app
        .active_file
        .as_ref()
        .map(|p| format!("Saved: {}", p.display()))
        .unwrap_or_default();

    let lines = vec![
        Line::from(Span::styled("Recording stopped.", bold(GREEN))),
        Line::from(Span::styled(file_msg, Style::default().fg(CYAN))),
        Line::from(Span::raw("")),
        Line::from(Span::styled(
            "Continue this chapter?",
            bold(WHITE),
        )),
        Line::from(Span::raw("")),
        Line::from(vec![
            Span::styled("  Y / Enter", bold(GREEN)),
            Span::styled(
                format!("  → record Part {:03} (same chapter)", app.session.part + 1),
                Style::default().fg(WHITE),
            ),
        ]),
        Line::from(vec![
            Span::styled("  N        ", bold(YELLOW)),
            Span::styled(
                format!(
                    "  → chapter {:02} complete, advance to Chapter {:02}",
                    app.session.chapter,
                    app.session.chapter + 1
                ),
                Style::default().fg(WHITE),
            ),
        ]),
    ];

    f.render_widget(Paragraph::new(lines), area);
}

// ── Punch rollback ────────────────────────────────────────────────────────────

fn draw_punch_rollback(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // ROLLING BACK banner
            Constraint::Length(1), // blank
            Constraint::Length(1), // playback counter
            Constraint::Length(1), // file being played
            Constraint::Length(1), // blank
            Constraint::Length(1), // abs timeline pos
            Constraint::Min(0),
        ])
        .split(area);

    // Banner
    let banner = Paragraph::new(Line::from(Span::styled(
        "  ◀◀  ROLLING BACK  ◀◀  ",
        Style::default()
            .fg(Color::Black)
            .bg(YELLOW)
            .add_modifier(Modifier::BOLD),
    )))
    .alignment(Alignment::Center);
    f.render_widget(banner, chunks[0]);

    // Playback counter
    let elapsed = app.playback_elapsed_secs();
    let total = app.punch_back_time.min(app.clip_start_timeline + app.last_clip_duration);
    let done_tag = if app.playback_done {
        Span::styled("  (end of clip)", dim())
    } else {
        Span::raw("")
    };
    f.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled("  Playback: ", dim()),
            Span::styled(
                format!("+{:.1} s / {:.1} s", elapsed, total),
                bold(WHITE),
            ),
            done_tag,
        ])),
        chunks[2],
    );

    // File being played
    if let Some(ref path) = app.active_file {
        f.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled("  File:     ", dim()),
                Span::styled(path.display().to_string(), Style::default().fg(CYAN)),
            ])),
            chunks[3],
        );
    }

    // Absolute timeline position of the rollback start
    f.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled("  Rollback: ", dim()),
            Span::styled(
                format!("{:.3} s (absolute)", app.punch_rollback_abs),
                Style::default().fg(DARK_GRAY),
            ),
        ])),
        chunks[5],
    );
}

// ── Mic error ─────────────────────────────────────────────────────────────────

fn draw_mic_error(f: &mut Frame, app: &App, area: Rect) {
    let border = Block::default()
        .borders(Borders::ALL)
        .border_style(bold(RED))
        .title(Span::styled(" MICROPHONE ERROR ", bold(RED)));

    let inner = inner_rect(area);
    f.render_widget(Clear, area);
    f.render_widget(border, area);

    let lines = vec![
        Line::from(Span::raw("")),
        Line::from(Span::styled(&app.error_msg, bold(RED))),
        Line::from(Span::raw("")),
        Line::from(Span::styled(
            "Connect the Deity VO-7U and press R to retry.",
            dim(),
        )),
    ];

    f.render_widget(
        Paragraph::new(lines).wrap(Wrap { trim: false }),
        inner,
    );
}

// ── Fatal error ───────────────────────────────────────────────────────────────

fn draw_fatal(f: &mut Frame, app: &App, area: Rect) {
    let border = Block::default()
        .borders(Borders::ALL)
        .border_style(bold(RED))
        .title(Span::styled(" FATAL ERROR ", bold(RED)));

    let inner = inner_rect(area);
    f.render_widget(Clear, area);
    f.render_widget(border, area);

    let lines = vec![
        Line::from(Span::raw("")),
        Line::from(Span::styled(&app.error_msg, bold(RED))),
        Line::from(Span::raw("")),
        Line::from(Span::styled("Press Q to exit.", dim())),
    ];

    f.render_widget(
        Paragraph::new(lines).wrap(Wrap { trim: false }),
        inner,
    );
}

// ── Footer ────────────────────────────────────────────────────────────────────

fn draw_footer(f: &mut Frame, app: &App, area: Rect) {
    let keys: &str = match app.mode {
        AppMode::Setup => "[Enter] Confirm   [Tab] Switch field   [Esc] Cancel",
        AppMode::Ready => "[Space] Start recording   [E] Edit session   [R] Re-detect mic   [Q] Quit",
        AppMode::PreCheck => "[Esc] Abort check",
        AppMode::Recording => "[Space] Stop   [P] Punch and roll",
        AppMode::PostRecording => "[Y/Enter] Continue chapter   [N] Chapter complete",
        AppMode::PunchRollback => "[Space] Punch in here   [Esc] Cancel",
        AppMode::MicError => "[R] Retry detection   [Q] Quit",
        AppMode::Fatal => "[Q] Quit",
    };

    f.render_widget(
        Paragraph::new(Span::styled(keys, dim())).alignment(Alignment::Center),
        area,
    );
}

// ── Helpers ───────────────────────────────────────────────────────────────────

const SPINNER: &[char] = &['|', '/', '-', '\\'];

/// Build a simple ASCII VU meter bar.
fn build_vu_meter(rms_db: f32, width: usize) -> String {
    if width == 0 {
        return String::new();
    }
    // Map [-60, 0] dBFS → [0, width]
    let fraction = ((rms_db - (-60.0)) / 60.0).clamp(0.0, 1.0) as f64;
    let filled = (fraction * width as f64) as usize;
    let mut bar = String::with_capacity(width + 2);
    bar.push('[');
    for i in 0..width {
        if i < filled {
            bar.push('█');
        } else {
            bar.push('░');
        }
    }
    bar.push(']');
    bar
}

/// Choose a colour for the level indicator based on dBFS value.
fn level_color(rms_db: f32) -> Color {
    if rms_db > -6.0 {
        RED
    } else if rms_db > -18.0 {
        YELLOW
    } else if rms_db > -60.0 {
        GREEN
    } else {
        DARK_GRAY
    }
}

/// Returns a rect shrunk by 1 on all sides (content inside a single-cell border).
fn inner_rect(r: Rect) -> Rect {
    Rect {
        x: r.x + 1,
        y: r.y + 1,
        width: r.width.saturating_sub(2),
        height: r.height.saturating_sub(2),
    }
}
