//! Local web server.
//!
//! Binds to `127.0.0.1:<BROWSER_PORT>` (default 7474).
//!
//! Routes:
//!   GET  /       — hello-world HTML (Phase 1; replaced by viewer in Phase 3)
//!   GET  /state  — current `BrowserState` as JSON
//!   WS   /ws     — real-time `BrowserState` push; one JSON frame per state change

pub mod state;

use std::sync::Arc;

use anyhow::Result;
use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    response::IntoResponse,
    routing::get,
    Router,
};
use tokio::sync::watch;

use state::{BrowserState, StateTx};

const DEFAULT_PORT: u16 = 7474;
const PORT_SEARCH_RANGE: u16 = 10;

// ── Spawn ─────────────────────────────────────────────────────────────────────

/// Bind a TCP listener on the configured port, trying up to
/// `PORT_SEARCH_RANGE` sequential ports if the preferred one is taken.
///
/// Returns `(bound_port, state_sender)`.  Push new `BrowserState` snapshots
/// into `state_sender` on every app tick; connected WebSocket clients and
/// `GET /state` callers will see them immediately.
pub fn spawn() -> Result<(u16, StateTx)> {
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

    // watch channel: the main loop sends snapshots; axum handlers read them.
    let (tx, rx) = watch::channel(BrowserState::default());
    let shared = Arc::new(rx);

    std::thread::spawn(move || {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("tokio runtime")
            .block_on(serve(std_listener, shared));
    });

    Ok((port, tx))
}

// ── Server ────────────────────────────────────────────────────────────────────

async fn serve(
    std_listener: std::net::TcpListener,
    rx: Arc<watch::Receiver<BrowserState>>,
) {
    let app = Router::new()
        .route("/", get(index))
        .route("/state", get(state_handler))
        .route("/ws", get(ws_handler))
        .with_state(rx);

    let listener = tokio::net::TcpListener::from_std(std_listener)
        .expect("convert std listener to tokio");

    if let Err(e) = axum::serve(listener, app).await {
        eprintln!("rustcorder: web server error: {e}");
    }
}

// ── Handlers ──────────────────────────────────────────────────────────────────

async fn index() -> axum::response::Html<&'static str> {
    axum::response::Html(
        r#"<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>rustcorder</title></head>
<body>
  <h1>rustcorder</h1>
  <p>Web UI coming soon (Phase 3).</p>
  <p>Current state: <a href="/state">/state</a></p>
  <p>WebSocket: <code>ws://localhost:PORT/ws</code></p>
</body>
</html>"#,
    )
}

async fn state_handler(
    State(rx): State<Arc<watch::Receiver<BrowserState>>>,
) -> axum::Json<BrowserState> {
    axum::Json(rx.borrow().clone())
}

async fn ws_handler(
    ws: WebSocketUpgrade,
    State(rx): State<Arc<watch::Receiver<BrowserState>>>,
) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_socket(socket, rx))
}

async fn handle_socket(mut socket: WebSocket, rx: Arc<watch::Receiver<BrowserState>>) {
    // Clone a new receiver so this connection has its own change cursor.
    let mut rx = (*rx).clone();

    // Send the current state immediately on connect.
    let initial = serde_json::to_string(&*rx.borrow()).unwrap_or_default();
    if socket.send(Message::Text(initial)).await.is_err() {
        return;
    }

    // Stream every subsequent change until the client disconnects.
    loop {
        tokio::select! {
            // New state available.
            result = rx.changed() => {
                if result.is_err() {
                    break; // sender dropped (process shutting down)
                }
                let json = serde_json::to_string(&*rx.borrow()).unwrap_or_default();
                if socket.send(Message::Text(json)).await.is_err() {
                    break; // client disconnected
                }
            }
            // Client sent something (close frame, ping, etc.).
            msg = socket.recv() => {
                match msg {
                    Some(Ok(Message::Close(_))) | None => break,
                    _ => {} // ignore other client messages for now (Phase 4 adds actions)
                }
            }
        }
    }
}
