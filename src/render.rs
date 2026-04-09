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

const RED: Color    = Color::Red;
const GREEN: Color  = Color::Green;
const YELLOW: Color = Color::Yellow;
const CYAN: Color   = Color::Cyan;
const WHITE: Color  = Color::White;
const DARK_GRAY: Color = Color::DarkGray;

fn bold(c: Color) -> Style { Style::default().fg(c).add_modifier(Modifier::BOLD) }
fn dim()          -> Style { Style::default().fg(DARK_GRAY) }

// ── Entry point ───────────────────────────────────────────────────────────────

pub fn draw(f: &mut Frame, app: &App) {
    let area = f.area();

    let outer = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(CYAN))
        .title(Span::styled(" RUSTCORDER ", bold(CYAN).add_modifier(Modifier::BOLD)));
    f.render_widget(outer, area);

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
        AppMode::Setup      => Span::styled("[ SETUP ]",          bold(YELLOW)),
        AppMode::Standby    => Span::styled("[ STANDBY ]",        bold(YELLOW)),
        AppMode::Playing    => Span::styled("[ ▶ PLAYING ]",      bold(GREEN)),
        AppMode::Recording  => Span::styled("[ ● RECORDING ]",    bold(RED)),
        AppMode::MicError   => Span::styled("[ MIC ERROR ]",      bold(RED)),
        AppMode::Fatal      => Span::styled("[ FATAL ERROR ]",    bold(RED)),
    };

    let book_label = if app.session.book.is_empty() {
        Span::styled("(no book)", dim())
    } else {
        Span::styled(app.session.book.clone(), bold(WHITE))
    };

    let sep = Span::styled("  │  ", dim());

    let pos_secs = app.playhead_pos();
    let pos_label = Span::styled(fmt_mm_ss(pos_secs), Style::default().fg(CYAN));

    let line = Line::from(vec![
        book_label,
        sep.clone(),
        Span::styled(format!("Ch {:02}", app.session.chapter), Style::default().fg(WHITE)),
        sep.clone(),
        Span::styled(format!("Part {:03}", app.session.part), Style::default().fg(WHITE)),
        sep.clone(),
        pos_label,
        Span::raw("  "),
        state_label,
    ]);

    let block = Block::default().borders(Borders::BOTTOM).border_style(dim());
    f.render_widget(Paragraph::new(line).block(block).alignment(Alignment::Left), area);
}

// ── Body ──────────────────────────────────────────────────────────────────────

fn draw_body(f: &mut Frame, app: &App, area: Rect) {
    match app.mode {
        AppMode::Setup     => draw_setup(f, app, area),
        AppMode::Standby   => draw_standby(f, app, area),
        AppMode::Playing   => draw_playing(f, app, area),
        AppMode::Recording => draw_recording(f, app, area),
        AppMode::MicError  => draw_mic_error(f, app, area),
        AppMode::Fatal     => draw_fatal(f, app, area),
    }
}

// ── Setup ─────────────────────────────────────────────────────────────────────

fn draw_setup(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Length(2),
            Constraint::Min(0),
        ])
        .split(area);

    f.render_widget(Paragraph::new("Session Setup").style(bold(CYAN)), chunks[0]);

    let book_style = if app.setup_field == SetupField::Book { bold(WHITE) } else { dim() };
    let book_cursor = if app.setup_field == SetupField::Book { "█" } else { "" };
    let book_block = Block::default()
        .borders(Borders::ALL)
        .border_style(if app.setup_field == SetupField::Book { bold(CYAN) } else { dim() })
        .title("Book name (used as directory)");
    f.render_widget(
        Paragraph::new(format!("{}{}", app.setup_book, book_cursor))
            .style(book_style).block(book_block),
        chunks[1],
    );

    let ch_style = if app.setup_field == SetupField::Chapter { bold(WHITE) } else { dim() };
    let ch_cursor = if app.setup_field == SetupField::Chapter { "█" } else { "" };
    let ch_block = Block::default()
        .borders(Borders::ALL)
        .border_style(if app.setup_field == SetupField::Chapter { bold(CYAN) } else { dim() })
        .title("Starting chapter number");
    f.render_widget(
        Paragraph::new(format!("{}{}", app.setup_chapter, ch_cursor))
            .style(ch_style).block(ch_block),
        chunks[2],
    );

    if let Some(ref err) = app.setup_error {
        f.render_widget(Paragraph::new(err.as_str()).style(bold(RED)), chunks[3]);
    }
}

// ── Standby ───────────────────────────────────────────────────────────────────

fn draw_standby(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1), // mic info
            Constraint::Length(1), // blank
            Constraint::Length(1), // next file
            Constraint::Length(1), // position
            Constraint::Length(1), // blank
            Constraint::Length(2), // status msg
            Constraint::Min(0),
        ])
        .split(area);

    let mic_line = match &app.mic {
        Some(m) => Line::from(vec![
            Span::styled("Microphone: ", dim()),
            Span::styled(m.description.clone(), bold(GREEN)),
            Span::styled("  ✓", bold(GREEN)),
        ]),
        None => Line::from(Span::styled("Microphone: NOT DETECTED", bold(RED))),
    };
    f.render_widget(Paragraph::new(mic_line), chunks[0]);

    let file_line = Line::from(vec![
        Span::styled("Next file:  ", dim()),
        Span::styled(
            app.session.current_path().display().to_string(),
            Style::default().fg(CYAN),
        ),
    ]);
    f.render_widget(Paragraph::new(file_line), chunks[2]);

    let pos_line = Line::from(vec![
        Span::styled("Position:   ", dim()),
        Span::styled(
            fmt_mm_ss(app.session.timeline_pos),
            bold(YELLOW),
        ),
    ]);
    f.render_widget(Paragraph::new(pos_line), chunks[3]);

    if let Some(ref msg) = app.status_msg {
        f.render_widget(
            Paragraph::new(msg.as_str()).style(bold(YELLOW)).wrap(Wrap { trim: true }),
            chunks[5],
        );
    }
}

// ── Playing ───────────────────────────────────────────────────────────────────

fn draw_playing(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // banner
            Constraint::Length(1), // blank
            Constraint::Length(1), // playhead position
            Constraint::Length(1), // origin
            Constraint::Min(0),
        ])
        .split(area);

    let banner = Paragraph::new(Line::from(Span::styled(
        "  ▶  PLAYING  ▶  ",
        Style::default().fg(Color::Black).bg(GREEN).add_modifier(Modifier::BOLD),
    )))
    .alignment(Alignment::Center);
    f.render_widget(banner, chunks[0]);

    let elapsed = app.playback_elapsed_secs();
    let playhead = app.playhead_origin + elapsed;
    let done_tag = if app.playback_done { Span::styled("  (end)", dim()) } else { Span::raw("") };

    f.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled("  Position: ", dim()),
            Span::styled(fmt_mm_ss(playhead), bold(GREEN)),
            done_tag,
        ])),
        chunks[2],
    );

    f.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled("  From:     ", dim()),
            Span::styled(fmt_mm_ss(app.playhead_origin), Style::default().fg(DARK_GRAY)),
        ])),
        chunks[3],
    );
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

    let banner = Paragraph::new(Line::from(Span::styled(
        "  ●  RECORDING  ●  ",
        Style::default().fg(Color::Black).bg(RED).add_modifier(Modifier::BOLD),
    )))
    .alignment(Alignment::Center);
    f.render_widget(banner, chunks[0]);

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

    if let Some(elapsed) = app.elapsed_recording() {
        let secs = elapsed.as_secs();
        let h = secs / 3600;
        let m = (secs % 3600) / 60;
        let s = secs % 60;
        f.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled("  Elapsed: ", dim()),
                Span::styled(format!("{:02}:{:02}:{:02}", h, m, s), bold(WHITE)),
            ])),
            chunks[5],
        );
    }

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
        Line::from(Span::styled("Connect the Deity VO-7U and press R to retry.", dim())),
    ];
    f.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }), inner);
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
    f.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }), inner);
}

// ── Footer ────────────────────────────────────────────────────────────────────

fn draw_footer(f: &mut Frame, app: &App, area: Rect) {
    let keys: &str = match app.mode {
        AppMode::Setup     => "[Enter] Confirm   [Tab] Switch field   [Esc] Cancel",
        AppMode::Standby   => "[Space] Record   [L] Play   [J] ◀ Part   [K] Part ▶   [N] Next chapter   [E] Edit   [Q] Quit",
        AppMode::Playing   => "[Space] Punch in   [L] Pause   [J] ◀ Part   [K] Part ▶   [P] Rewind",
        AppMode::Recording => "[Space] Stop   [P] Punch back",
        AppMode::MicError  => "[R] Retry detection   [Q] Quit",
        AppMode::Fatal     => "[Q] Quit",
    };

    f.render_widget(
        Paragraph::new(Span::styled(keys, dim())).alignment(Alignment::Center),
        area,
    );
}

// ── Helpers ───────────────────────────────────────────────────────────────────

const SPINNER: &[char] = &['|', '/', '-', '\\'];

/// Format seconds as `MM:SS.s`.
pub fn fmt_mm_ss(secs: f64) -> String {
    let total = secs.max(0.0);
    let m = (total / 60.0) as u64;
    let s = total % 60.0;
    format!("{:02}:{:04.1}", m, s)
}

fn build_vu_meter(rms_db: f32, width: usize) -> String {
    if width == 0 { return String::new(); }
    let fraction = ((rms_db - (-60.0)) / 60.0).clamp(0.0, 1.0) as f64;
    let filled = (fraction * width as f64) as usize;
    let mut bar = String::with_capacity(width + 2);
    bar.push('[');
    for i in 0..width {
        bar.push(if i < filled { '█' } else { '░' });
    }
    bar.push(']');
    bar
}

fn level_color(rms_db: f32) -> Color {
    if rms_db > -6.0       { RED }
    else if rms_db > -18.0 { YELLOW }
    else if rms_db > -60.0 { GREEN }
    else                   { DARK_GRAY }
}

fn inner_rect(r: Rect) -> Rect {
    Rect {
        x: r.x + 1,
        y: r.y + 1,
        width: r.width.saturating_sub(2),
        height: r.height.saturating_sub(2),
    }
}
