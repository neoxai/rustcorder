//! Local web server (Phase 1 skeleton).
//!
//! Binds to `127.0.0.1:<BROWSER_PORT>` (default 7474) and serves a minimal
//! HTML page at `GET /`.  Subsequent phases will add state broadcast,
//! the epub.js viewer, and WebSocket control.

use anyhow::Result;
use axum::{routing::get, Router};

const DEFAULT_PORT: u16 = 7474;
const PORT_SEARCH_RANGE: u16 = 10;

/// Bind a TCP listener on the configured port, trying up to
/// `PORT_SEARCH_RANGE` sequential ports if the preferred one is taken.
/// Returns the port that was successfully bound, or an error if all failed.
///
/// Binding happens synchronously here (before the tokio thread) so the
/// caller gets a concrete port — or a clear error — before proceeding.
pub fn spawn() -> Result<u16> {
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

    std::thread::spawn(move || {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("tokio runtime")
            .block_on(serve(std_listener));
    });

    Ok(port)
}

async fn serve(std_listener: std::net::TcpListener) {
    let app = Router::new().route("/", get(index));

    let listener = tokio::net::TcpListener::from_std(std_listener)
        .expect("convert std listener to tokio");

    if let Err(e) = axum::serve(listener, app).await {
        eprintln!("rustcorder: web server error: {e}");
    }
}

async fn index() -> axum::response::Html<&'static str> {
    axum::response::Html(
        r#"<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>rustcorder</title></head>
<body>
  <h1>rustcorder</h1>
  <p>Web UI coming soon (Phase 2+).</p>
</body>
</html>"#,
    )
}
