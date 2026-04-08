//! Local web server.
//!
//! Binds to `127.0.0.1:<BROWSER_PORT>` (default 7474).
//!
//! Routes:
//!   GET  /       — epub.js viewer (index.html)
//!   GET  /epub   — raw .epub file served to epub.js
//!   GET  /state  — current `BrowserState` as JSON (one-shot)
//!   WS   /ws     — real-time state push + incoming actions

pub mod state;

use std::sync::Arc;

use anyhow::Result;
use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    http::{header, StatusCode},
    response::IntoResponse,
    routing::get,
    Router,
};
use tokio::sync::watch;

use state::{ActionTx, ActionRx, BrowserState, StateTx, WebAction};

const DEFAULT_PORT: u16 = 7474;
const PORT_SEARCH_RANGE: u16 = 10;

// ── Shared server state ───────────────────────────────────────────────────────

struct ServerState {
    rx: watch::Receiver<BrowserState>,
    action_tx: ActionTx,
}

// ── Spawn ─────────────────────────────────────────────────────────────────────

/// Bind, spawn the server thread, and return:
/// - the bound port
/// - `StateTx` to push state snapshots from the app loop
/// - `ActionRx` to drain browser actions (epub_seek, etc.) from the app loop
pub fn spawn() -> Result<(u16, StateTx, ActionRx)> {
    let start = std::env::var("BROWSER_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_PORT);

    let (port, std_listener) = (start..start + PORT_SEARCH_RANGE)
        .find_map(|p| {
            std::net::TcpListener::bind(("127.0.0.1", p))
                .ok()
                .map(|l| (p, l))
        })
        .ok_or_else(|| {
            anyhow::anyhow!(
                "web server could not bind on ports {start}–{}",
                start + PORT_SEARCH_RANGE - 1
            )
        })?;

    std_listener.set_nonblocking(true)?;

    let (state_tx, state_rx) = watch::channel(BrowserState::default());
    // Sync channel with a small buffer so WebSocket handlers never block the
    // audio thread even if the main loop is briefly slow.
    let (action_tx, action_rx) = std::sync::mpsc::sync_channel::<WebAction>(32);

    let server_state = Arc::new(ServerState {
        rx: state_rx,
        action_tx,
    });

    std::thread::spawn(move || {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("tokio runtime")
            .block_on(serve(std_listener, server_state));
    });

    Ok((port, state_tx, action_rx))
}

// ── Server ────────────────────────────────────────────────────────────────────

async fn serve(
    std_listener: std::net::TcpListener,
    server_state: Arc<ServerState>,
) {
    let app = Router::new()
        .route("/", get(index))
        .route("/epub", get(epub_handler))
        .route("/state", get(state_handler))
        .route("/ws", get(ws_handler))
        .with_state(server_state);

    let listener = tokio::net::TcpListener::from_std(std_listener)
        .expect("convert std listener to tokio");

    if let Err(e) = axum::serve(listener, app).await {
        eprintln!("rustcorder: web server error: {e}");
    }
}

// ── Handlers ──────────────────────────────────────────────────────────────────

async fn index() -> axum::response::Html<&'static str> {
    axum::response::Html(include_str!("../../static/index.html"))
}

async fn epub_handler(
    State(ss): State<Arc<ServerState>>,
) -> impl IntoResponse {
    let path = ss.rx.borrow().epub_path.clone();
    match path {
        None => (StatusCode::NOT_FOUND, "No EPUB loaded").into_response(),
        Some(p) => match tokio::fs::read(&p).await {
            Ok(bytes) => (
                [(header::CONTENT_TYPE, "application/epub+zip")],
                bytes,
            )
                .into_response(),
            Err(e) => (
                StatusCode::NOT_FOUND,
                format!("Could not read EPUB: {e}"),
            )
                .into_response(),
        },
    }
}

async fn state_handler(
    State(ss): State<Arc<ServerState>>,
) -> axum::Json<BrowserState> {
    axum::Json(ss.rx.borrow().clone())
}

async fn ws_handler(
    ws: WebSocketUpgrade,
    State(ss): State<Arc<ServerState>>,
) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_socket(socket, ss))
}

// ── WebSocket connection ───────────────────────────────────────────────────────

async fn handle_socket(mut socket: WebSocket, ss: Arc<ServerState>) {
    let mut rx = ss.rx.clone();

    // Send current state immediately on connect.
    let initial = serde_json::to_string(&*rx.borrow()).unwrap_or_default();
    if socket.send(Message::Text(initial)).await.is_err() {
        return;
    }

    loop {
        // `biased` checks branches in declaration order.
        // socket.recv() is first so incoming browser messages are never
        // starved by the high-frequency state broadcasts (~25 Hz).
        tokio::select! {
            biased;

            // Incoming message from browser (highest priority).
            msg = socket.recv() => {
                match msg {
                    Some(Ok(Message::Text(text))) => {
                        handle_browser_message(&text, &ss.action_tx);
                    }
                    Some(Ok(Message::Close(_))) | None => break,
                    _ => {}
                }
            }

            // New state snapshot available — push to client.
            result = rx.changed() => {
                if result.is_err() { break; }
                let json = serde_json::to_string(&*rx.borrow()).unwrap_or_default();
                if socket.send(Message::Text(json)).await.is_err() { break; }
            }
        }
    }
}

/// Parse a JSON action message from the browser and forward it to the app loop.
fn handle_browser_message(text: &str, action_tx: &ActionTx) {
    let Ok(v) = serde_json::from_str::<serde_json::Value>(text) else {
        return;
    };
    match v.get("action").and_then(|a| a.as_str()) {
        Some("epub_seek") => {
            let cfi_str = v.get("cfi").and_then(|c| c.as_str()).unwrap_or("");
            if let Some(cfi) = crate::epub::EpubCfi::parse(cfi_str) {
                let _ = action_tx.try_send(WebAction::EpubSeek(cfi));
            }
        }
        // Recorder key actions — forwarded as WebAction::Key and converted to
        // synthetic KeyEvents in drain_actions.
        Some(name @ (
            "start" | "stop" |
            "punch" |
            "continue_chapter" | "chapter_complete" |
            "cancel" |
            "scroll_forward" | "scroll_back" |
            "retry_mic" | "edit_session" | "quit"
        )) => {
            let _ = action_tx.try_send(WebAction::Key(name.to_string()));
        }
        _ => {}
    }
}
