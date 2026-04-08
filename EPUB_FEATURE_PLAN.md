# EPUB Reader Feature — Implementation Plan

## Resolved decisions

| Question | Answer |
|---|---|
| EPUB source | Auto-discover the single `*.epub` in `<book>/` — no env var needed |
| Which modes show the pane | All modes (Ready, PreCheck, Recording, PostRecording, PunchRollback, MicError) |
| Auto-scroll when not Recording | Paused (only scrolls in Recording mode) |
| Scroll direction | `→` = forward, `←` = backward |
| Scroll during PunchRollback | Yes — `←`/`→` work freely in all modes |
| EPUB parsing | `epub` crate (Option A) |
| Layout | Bottom split pane (Option A) |
| Manual scroll unit | Lines (Option A) |
| Auto-scroll speed unit | Lines per minute (Option A) |

---

## Env variables

| Variable | Default | Meaning |
|---|---|---|
| `EPUB_SCROLL_LINES` | `1` | Lines scrolled per `→` / `←` keypress |
| `EPUB_AUTOSCROLL` | `0` | Auto-scroll speed in lines/min; `0` = off |
| `EPUB_PANE_RATIO` | `60` | % of body height given to the text pane |

---

## New files

| File | Purpose |
|---|---|
| `src/epub.rs` | EPUB loading, line-wrapping, scroll state, position save/load |
| `<book>/epub_position.txt` | Persisted scroll position (per book directory) |

---

## `epub_position.txt` format

Written to `<book>/epub_position.txt`, separate from the audio timeline:

```
EPUB=Beverly Cleary - Ralph 1 - The Mouse and the Motorcycle.epub
CHAR_OFFSET=48372
```

`CHAR_OFFSET` is the number of Unicode characters from the very start of the full
extracted plain text (all spine items concatenated, before any line-wrapping).
This is **layout-independent**: terminal resize re-wraps the text and re-derives
the scroll line from the stored char offset, so position is never lost on resize.

`CHAR_OFFSET` is the authoritative position.  The `scroll` field in `EpubReader`
is derived from it after wrapping and is re-derived any time the terminal width
changes.

Saved on every manual scroll keypress and every 5 seconds during auto-scroll.
Loaded on startup whenever the book directory is known.

---

## Module: `src/epub.rs`

```rust
pub struct EpubReader {
    /// Full plain text extracted from the EPUB (all spine items, tags stripped).
    /// This is the source of truth; `lines` is derived from it.
    full_text: String,
    /// `full_text` word-wrapped to the current pane width.
    /// Re-derived on every terminal resize.
    lines: Vec<String>,
    /// Char offset of the first character of `lines[scroll]` within `full_text`.
    /// This is the authoritative saved position — layout-independent.
    pub char_offset: usize,
    /// Index into `lines` derived from `char_offset`.  Re-derived after rewrap.
    scroll: usize,
    /// Path of the source EPUB file (for position save).
    epub_filename: String,
    /// Book directory (for position save path).
    book_dir: PathBuf,
    /// Char offset of the first char of each wrapped line (parallel to `lines`).
    /// Used to re-derive `scroll` from `char_offset` after a resize.
    line_offsets: Vec<usize>,
}

impl EpubReader {
    /// Find and load the single *.epub in `book_dir`.  Returns None if absent.
    pub fn load(book_dir: &Path, pane_width: u16) -> Result<Option<Self>>;

    /// Restore saved char_offset from epub_position.txt and re-derive scroll.
    pub fn restore_position(&mut self);

    /// Scroll forward/backward by `n` lines; updates scroll + char_offset.
    pub fn scroll_by(&mut self, delta: isize);

    /// Return the slice of lines visible in a pane of `height` rows.
    pub fn visible_lines(&self, height: usize) -> &[String];

    /// Re-wrap to a new pane width; re-derives scroll from char_offset.
    pub fn rewrap(&mut self, new_width: u16);

    /// Save current char_offset to epub_position.txt.
    pub fn save_position(&self) -> Result<()>;
}
```

**`char_offset` ↔ `scroll` contract:**
- `scroll_by(delta)`: clamp `scroll`, then set `char_offset = line_offsets[scroll]`.
- `rewrap(w)`: rebuild `lines` and `line_offsets` for new width, then set
  `scroll` to the last index where `line_offsets[i] <= char_offset`
  (binary search — always O(log n)).

**Text extraction from EPUB:**
1. Open the `.epub` (ZIP), parse `content.opf` for spine order.
2. For each spine item, read the HTML, strip all tags (simple char-scan — no HTML
   parser dep needed), collapse whitespace, split on paragraph boundaries.
3. Concatenate into `full_text` with `\n\n` between paragraphs.
4. Word-wrap `full_text` to `pane_width - 2` columns (greedy, no new dep);
   record `line_offsets` in parallel.
5. Store `lines` + `line_offsets`.

---

## Changes to `App` struct (`src/app.rs`)

```rust
// ── EPUB reader ───────────────────────────────────────────────────────────
pub epub: Option<EpubReader>,
/// Lines to scroll per keypress.  From EPUB_SCROLL_LINES (default 1).
pub epub_scroll_lines: usize,
/// Auto-scroll speed in lines/min.  0 = disabled.  From EPUB_AUTOSCROLL.
pub epub_autoscroll_lpm: f64,
/// Fractional-line accumulator for sub-line auto-scroll precision.
pub epub_autoscroll_accum: f64,
/// Seconds since last auto-save of epub_position.txt.
pub epub_autosave_accum: f64,
/// Last terminal pane width seen; used to trigger rewrap on resize.
pub epub_pane_width: u16,
```

**On startup / book change:** call `EpubReader::load(&session.output_dir(), pane_width)`,
then `restore_position()`.

---

## Key handling (`src/app.rs`)

`←` / `→` are handled **before** the per-mode dispatch — they always fire regardless
of current mode:

```rust
pub fn handle_key(&mut self, key: KeyEvent, pane_width: u16) -> bool {
    match key.code {
        KeyCode::Left => {
            if let Some(ref mut epub) = self.epub {
                epub.scroll_by(-(self.epub_scroll_lines as isize));
                let _ = epub.save_position();
            }
            return false;
        }
        KeyCode::Right => {
            if let Some(ref mut epub) = self.epub {
                epub.scroll_by(self.epub_scroll_lines as isize);
                let _ = epub.save_position();
            }
            return false;
        }
        _ => {}
    }
    // ... existing per-mode dispatch
}
```

---

## Auto-scroll tick (`src/app.rs` → `tick()`)

Only advances when `self.mode == AppMode::Recording`:

```rust
// Auto-scroll — only while actively recording.
if self.mode == AppMode::Recording {
    if self.epub_autoscroll_lpm > 0.0 {
        if let Some(ref mut epub) = self.epub {
            let lines_per_tick = self.epub_autoscroll_lpm / 60.0 * TICK_SECS;
            self.epub_autoscroll_accum += lines_per_tick;
            if self.epub_autoscroll_accum >= 1.0 {
                let steps = self.epub_autoscroll_accum as isize;
                epub.scroll_by(steps);
                self.epub_autoscroll_accum -= steps as f64;
            }
            self.epub_autosave_accum += TICK_SECS;
            if self.epub_autosave_accum >= 5.0 {
                let _ = epub.save_position();
                self.epub_autosave_accum = 0.0;
            }
        }
    }
}
```

---

## Layout changes (`src/render.rs`)

`draw_body()` checks whether an EPUB is loaded.  If yes, it splits vertically:

```
body area
├── top (100 - EPUB_PANE_RATIO)%  →  existing draw_*() function
└── bottom EPUB_PANE_RATIO%       →  new draw_epub_pane()
```

`draw_epub_pane()`:
- Bordered block titled `" EPUB "` with filename and `line N / total`.
- Calls `epub.visible_lines(inner_height)` and renders each as a `Line`.
- Current top line is slightly highlighted (dim underline) to show position.
- Shows `[auto-scroll]` tag in the title when `epub_autoscroll_lpm > 0` and mode
  is Recording.

If no EPUB is loaded, body renders exactly as it does today (no change).

---

## Resize handling

`pane_width` is passed from the render loop into any place that calls
`epub.visible_lines()`.  When `f.area()` changes width, call `epub.rewrap(new_width)`.
Scroll index is preserved (clamped if the new wrap produces fewer lines).

---

## Implementation order

1. `src/epub.rs` — `EpubReader::load`, `scroll_by`, `visible_lines`, `save_position`,
   `restore_position`, `rewrap`
2. `Cargo.toml` — add `epub = "2"`
3. `src/app.rs` — add fields, load on startup/book-change, wire `←`/`→`, add
   auto-scroll to `tick()`
4. `src/render.rs` — split body pane, `draw_epub_pane()`
5. `.env.example` — document `EPUB_SCROLL_LINES`, `EPUB_AUTOSCROLL`, `EPUB_PANE_RATIO`
