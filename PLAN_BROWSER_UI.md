# Browser EPUB Viewer — Implementation Plan

## Overview

The terminal EPUB pane renders plain stripped text, which loses formatting, font sizing, and comfortable reading layout. This plan replaces it with a full browser-based epub.js viewer running on a local web server that the Rust process spawns at startup.

The browser window serves as a **second screen**: the narrator reads along in the browser while interacting with the recorder from either the browser or the terminal. Both UIs stay in sync in real-time via WebSocket.

---

## Position Representation: CFI Everywhere

**epub.js uses EPUB CFI (Canonical Fragment Identifier) as its native position format.**
The entire program — on-disk storage, wire protocol, and the Rust internal state — will adopt CFI as the single authoritative position type, replacing the current `char_offset: usize` approach.

### Why CFI instead of char offset

| | `char_offset` (current) | CFI (new) |
|---|---|---|
| epub.js compatible | No — requires conversion | Yes — native format |
| Survives text extraction changes | No — strip_html tweaks shift all offsets | Yes — tied to document structure |
| Human-readable | Barely | Somewhat (spine index + offset) |
| Rust-parseable | Trivially | Requires a small parser |
| On-disk format | `CHAR_OFFSET=48372` | `CFI=epubcfi(/6/4!/1:48372)` |

### CFI format used in this project

Full CFI strings can be deeply nested. We use a **simplified spine-item CFI** that is both valid for epub.js and trivially parseable in Rust:

```
epubcfi(/6/N!/1:C)
```

- `/6` — the EPUB `<spine>` element (always element 6 in the OPF package)
- `/N` — even-numbered index of the spine item (2 = first item, 4 = second, …)
- `!` — indirection into that spine item's document
- `/1:C` — character offset `C` within the document's root text node

This is sufficient for our use case (linear text, no nested element navigation). epub.js accepts it and will scroll to the correct position. Rust only needs to parse two integers: `N` and `C`.

### Rust internal representation

```rust
/// Canonical position within an EPUB. All position state in the program
/// uses this type; `char_offset` is never stored directly.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct EpubCfi {
    /// Even-numbered spine child index (2 = first spine item, 4 = second, …).
    pub spine_child: u32,
    /// Character offset within that spine item's extracted plain text.
    pub char_offset: u32,
}

impl EpubCfi {
    /// Serialize to the canonical string form used in epub_position.txt and
    /// JSON wire messages.
    pub fn to_string(&self) -> String {
        format!("epubcfi(/6/{}!/1:{})", self.spine_child, self.char_offset)
    }

    /// Parse from `epubcfi(/6/N!/1:C)`. Returns None on any parse failure.
    pub fn parse(s: &str) -> Option<Self> { … }
}
```

### Mapping CFI ↔ terminal scroll line

`EpubReader` must track per-spine-item boundaries so it can convert between a `EpubCfi` and the flat `scroll` line index used for terminal display:

```rust
pub struct EpubReader {
    full_text: String,
    lines: Vec<String>,
    line_offsets: Vec<usize>,          // char offsets into full_text (unchanged)

    /// Absolute char offset in `full_text` of the first character of each
    /// spine item. Length == number of spine items.
    spine_starts: Vec<usize>,

    /// CFI is now the authoritative saved position (replaces char_offset).
    pub cfi: EpubCfi,

    /// Scroll line derived from cfi — always kept in sync, never saved.
    pub scroll: usize,

    epub_filename: String,
    book_dir: PathBuf,
}
```

Conversion helpers (private to `epub.rs`):

```rust
fn cfi_to_flat_offset(&self, cfi: &EpubCfi) -> usize {
    let item_idx = (cfi.spine_child / 2).saturating_sub(1) as usize;
    let base = self.spine_starts.get(item_idx).copied().unwrap_or(0);
    base + cfi.char_offset as usize
}

fn flat_offset_to_cfi(&self, flat: usize) -> EpubCfi {
    // Find last spine_starts entry <= flat.
    let item_idx = self.spine_starts
        .partition_point(|&s| s <= flat)
        .saturating_sub(1);
    EpubCfi {
        spine_child: (item_idx as u32 + 1) * 2,
        char_offset: (flat - self.spine_starts[item_idx]) as u32,
    }
}
```

`scroll_by()` updates `scroll` first (as it does today), then re-derives `cfi` via `flat_offset_to_cfi`.

### `epub_position.txt` new format

```
EPUB=Beverly Cleary - Ralph 1.epub
CFI=epubcfi(/6/4!/1:48372)
```

`CHAR_OFFSET` is removed. The Rust parser for this file becomes a two-key reader. Old files with `CHAR_OFFSET` are silently ignored (position resets to beginning), which is the same behaviour as a mismatched EPUB filename today.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  rustcorder process                                     │
│                                                         │
│  ┌──────────────┐   shared Arc<Mutex<AppState>>         │
│  │  TUI thread  │◄──────────────────────────────────┐   │
│  │  (existing)  │                                   │   │
│  └──────────────┘                                   │   │
│                                                     │   │
│  ┌──────────────────────────────────────────────┐   │   │
│  │  Web server thread (axum, port 7474)         │   │   │
│  │                                              │   │   │
│  │  GET  /            → viewer HTML             │   │   │
│  │  GET  /epub        → serve raw .epub file    │   │   │
│  │  GET  /state       → current state JSON      │   │   │
│  │  POST /action      → trigger recorder action │   │   │
│  │  WS   /ws          → real-time state push    │   │   │
│  └─────────────────────────────┬────────────────┘   │   │
│                                │                    │   │
└────────────────────────────────┼────────────────────┘   │
                                 │                        │
                    WebSocket (JSON frames)               │
                                 │                        │
              ┌──────────────────▼──────────────────┐     │
              │  Browser  (localhost:7474)           │     │
              │                                     │     │
              │  ┌─────────────────────────────┐    │     │
              │  │  Status bar (mirrors TUI)   │    │     │
              │  │  Mode · Book · Chapter · VU │    │     │
              │  └─────────────────────────────┘    │     │
              │  ┌─────────────────────────────┐    │     │
              │  │  epub.js rendition          │    │     │
              │  │  (loaded to saved CFI pos.) │    │     │
              │  └─────────────────────────────┘    │     │
              └─────────────────────────────────────┘     │
```

---

## Web Server

### Technology

Add `axum` (with `tokio`) as the HTTP/WebSocket layer. Axum is a good fit because:
- It has first-class WebSocket support in the same crate, no extra WS library needed.
- Tower middleware makes serving static files and JSON trivial.
- The runtime is `tokio`, which plays well with the existing `std::thread`-based design via `tokio::runtime::Runtime::block_on`.

### Port

Default: **7474**. Configurable via `.env`:

```
BROWSER_PORT=7474
BROWSER_OPEN=true   # auto-open system browser on startup
```

### Startup

In `main.rs`, after the app is initialized but before the TUI event loop, spawn a `std::thread` that owns a `tokio` runtime and runs the axum server. Pass a `Arc<Mutex<App>>` (or a purpose-built `Arc<Mutex<BrowserState>>` — see below) to the server thread.

**Prefer a separate `BrowserState` struct** rather than sharing the full `App`, to avoid lock contention on the hot audio path. The TUI event loop updates `BrowserState` at the same tick rate it already redraws the TUI (~10 Hz).

---

## Shared State (`BrowserState`)

Minimal snapshot, cheaply cloneable, JSON-serialisable:

```rust
#[derive(Clone, serde::Serialize)]
pub struct BrowserState {
    pub mode: String,               // "Ready", "Recording", "PunchRollback", …
    pub book: String,
    pub chapter: u32,
    pub part: u32,
    pub elapsed_secs: f64,          // 0.0 when not recording
    pub rms_dbfs: f32,              // current level
    pub silence_warning: bool,
    pub epub_path: Option<String>,  // relative path to .epub file for /epub endpoint
    pub epub_cfi: Option<String>,   // current position as CFI string, or None if no EPUB
    pub timeline_pos_secs: f64,
    pub footer_hints: String,       // key hint line from the TUI footer
}
```

`epub_cfi` is the CFI string (e.g. `"epubcfi(/6/4!/1:48372)"`) — no conversion needed on either end. epub.js consumes it directly; Rust produces it directly from `EpubReader::cfi.to_string()`.

The WebSocket broadcaster sends a serialised `BrowserState` JSON frame to all connected clients whenever the state changes. Use `tokio::sync::broadcast` channel (capacity ~4) so the web thread never blocks the audio thread.

---

## API Endpoints

### `GET /`

Returns `index.html` (embedded in the binary via `include_str!` or served from a `static/` directory). The page loads epub.js from a CDN or from a bundled file under `static/`.

### `GET /epub`

Streams the raw `.epub` file from disk. Required because epub.js in the browser needs to fetch the file bytes directly. Content-Type: `application/epub+zip`.

### `GET /state`

Returns current `BrowserState` as JSON. Used for initial page load before the WebSocket connects.

### `POST /action`

Body: `{ "action": "start" | "stop" | "punch" | "scroll_forward" | "scroll_back" }`.

The handler sends an `AppEvent` into the existing `mpsc` channel that the TUI event loop already reads. This means **no new code paths in the state machine** — the browser just injects synthetic key events exactly as if the user pressed a key in the terminal.

### `WS /ws`

On connect: immediately send current `BrowserState` JSON.  
Ongoing: broadcast `BrowserState` JSON whenever state changes (~10 Hz max).  
Incoming messages: same `action` JSON as `POST /action` (lets the frontend use one code path for both WebSocket and REST). Includes the position update action:

```json
{ "action": "epub_seek", "cfi": "epubcfi(/6/6!/1:1203)" }
```

No conversion required — Rust parses this directly with `EpubCfi::parse()` and calls `EpubReader::seek_to_cfi()`.

---

## Frontend

Single HTML file (`static/index.html`) with inline JS. No build step required.

### Layout

```
┌────────────────────────────────────────────────────────┐
│ STATUS BAR                                             │
│  [● RECORDING]  Book: ...  Ch 03  Part 002  00:04:12  │
│  VU: ████████░░░░  -18.3 dBFS                         │
├────────────────────────────────────────────────────────┤
│                                                        │
│   epub.js rendition (fills remaining height)           │
│   — continuous scroll layout recommended               │
│   — font size/family configurable by user              │
│                                                        │
└────────────────────────────────────────────────────────┘
```

The status bar mirrors the terminal header + recording pane. Use the same color logic:
- Red background when `mode === "Recording"`.
- Amber VU bar with green/yellow/red coloring based on dBFS thresholds.

### epub.js Integration

Because CFI is epub.js's native format, the integration is now straightforward — no location generation, no offset conversion:

```javascript
const book = ePub("/epub");
const rendition = book.renderTo("viewer", {
    flow: "scrolled-doc",
    width: "100%",
    height: "100%"
});

// Navigate to saved position on load — CFI passed directly
const state = await fetch("/state").then(r => r.json());
if (state.epub_cfi) {
    rendition.display(state.epub_cfi);
}

// Send position back to Rust when the user scrolls
rendition.on("relocated", location => {
    ws.send(JSON.stringify({
        action: "epub_seek",
        cfi: location.start.cfi
    }));
});
```

No `charOffsetToCfi()` helper needed. No `book.locations.generate()` call needed (which is slow on large books). The CFI from `location.start.cfi` is passed back to Rust as-is.

### Keyboard Bindings (browser)

| Key | Action sent to Rust |
|-----|---------------------|
| Space / Enter | `start` or `stop` (depends on mode) |
| P | `punch` |
| → / L | `scroll_forward` |
| ← / H | `scroll_back` |
| Y | `continue_chapter` |
| N | `chapter_complete` |
| Escape | `cancel` |

Intercept these in a `keydown` listener; `preventDefault()` to stop the browser from scrolling on Space. Display a small key-hint bar below the status bar (populated from `state.footer_hints`).

---

## Position Synchronisation

CFI is the position currency on both sides, so synchronisation is now symmetric:

**Rust → Browser**: `BrowserState.epub_cfi` carries the current CFI string in every broadcast. The browser calls `rendition.display(cfi)` only if the incoming CFI differs significantly from the current viewport position (compare spine_child index to avoid fighting user's fine scroll).

**Browser → Rust**: The `relocated` event fires a `{ action: "epub_seek", cfi: "..." }` WebSocket message. Rust calls `EpubCfi::parse()`, then `EpubReader::seek_to_cfi()` which converts to a flat offset via `cfi_to_flat_offset()`, updates `scroll`, and saves the position file.

There is no impedance mismatch — both sides speak the same format.

### Terminal pane during browser session

Add an env var `EPUB_PANE=terminal|browser|both` (default `terminal`). When set to `browser`, suppress the terminal EPUB pane entirely and reclaim the space for the recording meters. The browser window is the reader; the terminal is the recorder. This is the cleanest division of concerns.

---

## Implementation Phases

### Phase 0 — Refactor `epub.rs` to use CFI (prerequisite, no web code)

This phase stands alone and can be done before any web work begins.

- Add `EpubCfi` struct with `to_string()` and `parse()` to `epub.rs`.
- Extend `extract_text()` to return `(String, Vec<usize>)` — the full text plus `spine_starts` offsets.
- Add `spine_starts: Vec<usize>` field to `EpubReader`.
- Replace `pub char_offset: usize` with `pub cfi: EpubCfi`.
- Add `cfi_to_flat_offset()` and `flat_offset_to_cfi()` private helpers.
- Update `scroll_by()`: after updating `scroll`, re-derive `cfi` via `flat_offset_to_cfi`.
- Add `seek_to_cfi()`: parse CFI → flat offset → update `scroll` and `cfi`.
- Update `save_position()` to write `CFI=` instead of `CHAR_OFFSET=`.
- Update `restore_position()` to parse `CFI=` (silently skip `CHAR_OFFSET=` lines).
- Update all callers in `app.rs` that read `epub.char_offset` directly.
- All existing tests must pass; add a test for CFI round-trip.

### Phase 1 — Web server skeleton (no UI yet)
- Add `axum`, `tokio`, `serde`, `serde_json` to `Cargo.toml`.
- Create `src/web/mod.rs` with a minimal axum router.
- Spawn server thread in `main.rs`; confirm `localhost:7474` responds.
- Serve a static "hello world" page.

### Phase 2 — State broadcast
- Define `BrowserState` struct (with `epub_cfi: Option<String>`) in `src/web/state.rs`.
- Add `tokio::sync::broadcast::Sender<BrowserState>` to `App` (or a wrapper).
- TUI event loop sends a snapshot on every state transition.
- Wire up `GET /state` and `WS /ws`.

### Phase 3 — EPUB viewer
- Implement `GET /epub` endpoint.
- Write `static/index.html` with epub.js (load from CDN first, bundle later).
- Display epub at `state.epub_cfi` on load — direct `rendition.display(cfi)` call.
- Wire `rendition.on("relocated")` → WebSocket `epub_seek` → `EpubReader::seek_to_cfi()`.

### Phase 4 — Status bar + key bindings
- Add status bar HTML/CSS mirroring the TUI header and recording pane.
- Connect WebSocket state updates to DOM updates (mode, elapsed, VU meter).
- Add `keydown` listener with full key mapping.
- Test punch-and-roll round-trip from browser keyboard.

### Phase 5 — Polish
- Add `EPUB_PANE` env var; suppress terminal EPUB pane when set to `browser`.
- Auto-open browser via `xdg-open` / `open` when `BROWSER_OPEN=true`.
- Embed `index.html` in the binary via `include_str!` so there are no external file dependencies.
- Bundle epub.js locally (avoid CDN dependency for offline use).

---

## New Dependencies

```toml
axum       = { version = "0.7", features = ["ws"] }
tokio      = { version = "1",   features = ["rt-multi-thread", "macros"] }
serde      = { version = "1",   features = ["derive"] }
serde_json = "1"
tower-http = { version = "0.5", features = ["fs"] }   # optional: static file serving
```

`tokio` may already be pulled in transitively; check `cargo tree` first.

---

## Open Questions

1. **CFI precision at spine boundaries**: When the user is at the very last character of a spine item, `flat_offset_to_cfi` must not spill over into the next item. The conversion should clamp `char_offset` to `spine_item_len - 1`.

2. **EPUB DRM**: The `epub` crate (Rust side) only handles unencrypted EPUBs, and epub.js in the browser also only handles unencrypted files. This is fine for personal audiobook narration use.

3. **Multiple browser tabs**: The broadcast channel handles multiple WebSocket clients naturally. Seek actions from any tab are accepted; last writer wins for position. Acceptable for a single-user local tool.

4. **Security**: The server binds to `127.0.0.1` only, not `0.0.0.0`. No authentication needed for a local tool.
