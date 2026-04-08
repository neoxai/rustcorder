//! EPUB loading, text extraction, line-wrapping, and scroll-position management.
//!
//! Position is tracked as a **character offset** into the full extracted plain
//! text (all spine items concatenated, before wrapping).  This is
//! layout-independent: the terminal can be resized at any time and
//! `rewrap()` will restore the correct scroll line from the saved offset.

use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use anyhow::Result;
use epub::doc::EpubDoc;

// ── Public types ──────────────────────────────────────────────────────────────

pub struct EpubReader {
    /// Full plain text extracted from the EPUB (spine order, tags stripped).
    /// This is the source of truth; `lines` is always derived from it.
    full_text: String,
    /// `full_text` word-wrapped to the current pane width.
    lines: Vec<String>,
    /// `line_offsets[i]` = char offset in `full_text` of `lines[i]`'s first char.
    line_offsets: Vec<usize>,
    /// Char offset of the first char of the top visible line.
    /// **Authoritative saved position** — survives terminal resize.
    pub char_offset: usize,
    /// Index into `lines` / `line_offsets` for the top visible line.
    /// Always derived from `char_offset`; re-derived after every `rewrap`.
    pub scroll: usize,
    /// Filename of the source EPUB (stored in epub_position.txt).
    epub_filename: String,
    /// Book directory (used for epub_position.txt path).
    book_dir: PathBuf,
}

impl EpubReader {
    // ── Construction ─────────────────────────────────────────────────────

    /// Scan `book_dir` for the single `*.epub` file, extract its text, and
    /// wrap it to `pane_width` columns.  Returns `None` if no EPUB is found.
    pub fn load(book_dir: &Path, pane_width: u16) -> Result<Option<Self>> {
        let epub_path = match find_epub(book_dir)? {
            Some(p) => p,
            None => return Ok(None),
        };

        let epub_filename = epub_path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();

        let full_text = extract_text(&epub_path)?;
        let width = inner_width(pane_width);
        let (lines, line_offsets) = wrap_text(&full_text, width);

        Ok(Some(EpubReader {
            full_text,
            lines,
            line_offsets,
            char_offset: 0,
            scroll: 0,
            epub_filename,
            book_dir: book_dir.to_path_buf(),
        }))
    }

    // ── Position persistence ──────────────────────────────────────────────

    /// Load `epub_position.txt` from the book directory and restore
    /// `char_offset` + `scroll`.  Silently no-ops if the file is missing or
    /// belongs to a different EPUB.
    pub fn restore_position(&mut self) {
        let path = self.position_path();
        let text = match fs::read_to_string(&path) {
            Ok(t) => t,
            Err(_) => return,
        };

        let mut saved_epub = String::new();
        let mut saved_offset: Option<usize> = None;

        for line in text.lines() {
            if let Some(v) = line.strip_prefix("EPUB=") {
                saved_epub = v.trim().to_string();
            } else if let Some(v) = line.strip_prefix("CHAR_OFFSET=") {
                saved_offset = v.trim().parse().ok();
            }
        }

        // Only restore if the EPUB filename matches.
        if saved_epub != self.epub_filename {
            return;
        }
        if let Some(offset) = saved_offset {
            self.char_offset = offset.min(self.full_text.chars().count());
            self.scroll = self.scroll_for_offset(self.char_offset);
        }
    }

    /// Write current position to `<book>/epub_position.txt`.
    pub fn save_position(&self) -> Result<()> {
        let path = self.position_path();
        let mut f = fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)?;
        writeln!(f, "EPUB={}", self.epub_filename)?;
        writeln!(f, "CHAR_OFFSET={}", self.char_offset)?;
        Ok(())
    }

    // ── Scrolling ─────────────────────────────────────────────────────────

    /// Scroll forward (`delta > 0`) or backward (`delta < 0`) by `|delta|` wrapped
    /// lines.  Updates both `scroll` and `char_offset`.
    pub fn scroll_by(&mut self, delta: isize) {
        let max = self.lines.len().saturating_sub(1);
        self.scroll = (self.scroll as isize + delta).max(0).min(max as isize) as usize;
        self.char_offset = self.line_offsets[self.scroll];
    }

    // ── Text access ───────────────────────────────────────────────────────

    /// Slice of wrapped lines starting at `scroll`, up to `height` lines.
    pub fn visible_lines(&self, height: usize) -> &[String] {
        let end = (self.scroll + height).min(self.lines.len());
        &self.lines[self.scroll..end]
    }

    /// Total number of wrapped lines (for progress display).
    pub fn total_lines(&self) -> usize {
        self.lines.len()
    }

    /// Filename of the loaded EPUB (for display in the pane title).
    pub fn epub_filename(&self) -> &str {
        &self.epub_filename
    }

    // ── Resize ────────────────────────────────────────────────────────────

    /// Re-wrap all lines to a new pane width.  `scroll` is re-derived from
    /// `char_offset` so the same passage stays at the top of the pane.
    pub fn rewrap(&mut self, new_pane_width: u16) {
        let width = inner_width(new_pane_width);
        let (lines, line_offsets) = wrap_text(&self.full_text, width);
        self.lines = lines;
        self.line_offsets = line_offsets;
        self.scroll = self.scroll_for_offset(self.char_offset);
    }

    // ── Helpers ───────────────────────────────────────────────────────────

    fn position_path(&self) -> PathBuf {
        self.book_dir.join("epub_position.txt")
    }

    /// Binary-search `line_offsets` for the last entry whose offset ≤ `target`.
    fn scroll_for_offset(&self, target: usize) -> usize {
        // `partition_point` finds the first index where the predicate is false.
        self.line_offsets
            .partition_point(|&o| o <= target)
            .saturating_sub(1)
    }
}

// ── EPUB discovery ────────────────────────────────────────────────────────────

fn find_epub(book_dir: &Path) -> Result<Option<PathBuf>> {
    if !book_dir.exists() {
        return Ok(None);
    }
    for entry in fs::read_dir(book_dir)? {
        let path = entry?.path();
        if path.extension().and_then(|e| e.to_str()) == Some("epub") {
            return Ok(Some(path));
        }
    }
    Ok(None)
}

// ── Text extraction ───────────────────────────────────────────────────────────

fn extract_text(epub_path: &Path) -> Result<String> {
    let mut doc = EpubDoc::new(epub_path)
        .map_err(|e| anyhow::anyhow!("EPUB open failed: {}", e))?;

    let n = doc.get_num_chapters();
    let mut full = String::new();

    for _ in 0..n {
        if let Some((html, mime)) = doc.get_current_str() {
            // Skip non-HTML spine items (images, CSS, etc.)
            if mime.contains("html") || mime.contains("xhtml") {
                let text = strip_html(&html);
                let trimmed = text.trim();
                if !trimmed.is_empty() {
                    if !full.is_empty() {
                        full.push('\n');
                    }
                    full.push_str(trimmed);
                }
            }
        }
        doc.go_next();
    }

    Ok(full)
}

// ── HTML stripping ────────────────────────────────────────────────────────────

fn strip_html(html: &str) -> String {
    let mut out = String::with_capacity(html.len());
    let mut chars = html.chars().peekable();
    let mut in_tag = false;
    let mut tag_buf = String::new();
    // Track script/style blocks so their content is skipped entirely.
    let mut skip_depth: usize = 0;

    while let Some(ch) = chars.next() {
        match ch {
            '<' => {
                in_tag = true;
                tag_buf.clear();
            }
            '>' if in_tag => {
                in_tag = false;
                let lower = tag_buf.trim().to_ascii_lowercase();
                let name = lower.trim_start_matches('/').split_whitespace().next().unwrap_or("");

                // Enter/leave script and style blocks.
                if name == "script" || name == "style" {
                    if lower.starts_with('/') { skip_depth = skip_depth.saturating_sub(1); }
                    else { skip_depth += 1; }
                } else if lower.starts_with("script") || lower.starts_with("style") {
                    skip_depth += 1;
                }

                if skip_depth > 0 {
                    // skip
                } else {
                    // Block-level tags → insert a newline to separate paragraphs.
                    let is_block = matches!(
                        name,
                        "p" | "h1" | "h2" | "h3" | "h4" | "h5" | "h6"
                            | "li" | "dt" | "dd"
                            | "div" | "section" | "article" | "blockquote"
                            | "tr" | "th" | "td"
                            | "br" | "hr"
                    );
                    if is_block && !out.ends_with('\n') {
                        out.push('\n');
                    }
                }
                tag_buf.clear();
            }
            _ if in_tag => {
                tag_buf.push(ch);
            }
            '&' if !in_tag && skip_depth == 0 => {
                // Collect entity up to ';', max 10 chars to avoid runaway.
                let mut entity = String::from('&');
                for _ in 0..12 {
                    match chars.peek() {
                        Some(&';') => {
                            chars.next();
                            entity.push(';');
                            break;
                        }
                        Some(&c) if c.is_alphanumeric() || c == '#' => {
                            entity.push(c);
                            chars.next();
                        }
                        _ => break,
                    }
                }
                out.push_str(&decode_entity(&entity));
            }
            _ if !in_tag && skip_depth == 0 => {
                out.push(ch);
            }
            _ => {}
        }
    }

    // Collapse runs of whitespace on each line; collapse 3+ newlines → 2.
    normalize_whitespace(&out)
}

fn decode_entity(e: &str) -> String {
    match e {
        "&amp;"   => "&",
        "&lt;"    => "<",
        "&gt;"    => ">",
        "&quot;"  => "\"",
        "&apos;"  => "'",
        "&nbsp;"  => " ",
        "&mdash;" => "—",
        "&ndash;" => "–",
        "&lsquo;" => "\u{2018}",
        "&rsquo;" => "\u{2019}",
        "&ldquo;" => "\u{201C}",
        "&rdquo;" => "\u{201D}",
        "&hellip;"=> "…",
        "&copy;"  => "©",
        "&reg;"   => "®",
        "&trade;" => "™",
        _ => {
            // Numeric entities: &#NNN; or &#xHHH;
            let inner = e.trim_start_matches('&').trim_end_matches(';');
            if let Some(hex) = inner.strip_prefix("#x").or_else(|| inner.strip_prefix("#X")) {
                if let Some(c) = u32::from_str_radix(hex, 16).ok().and_then(char::from_u32) {
                    return c.to_string();
                }
            } else if let Some(dec) = inner.strip_prefix('#') {
                if let Some(c) = dec.parse::<u32>().ok().and_then(char::from_u32) {
                    return c.to_string();
                }
            }
            return e.to_string(); // unknown — pass through
        }
    }
    .to_string()
}

fn normalize_whitespace(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut prev_newline_count = 0usize;
    let mut in_space = false;

    for ch in s.chars() {
        match ch {
            '\n' => {
                if in_space {
                    // flush any pending space before the newline
                    in_space = false;
                }
                if prev_newline_count < 2 {
                    out.push('\n');
                }
                prev_newline_count += 1;
            }
            ' ' | '\t' | '\r' => {
                if prev_newline_count == 0 {
                    in_space = true; // collapse to single space later
                }
                // Whitespace after newline is dropped (leading indent removed).
            }
            _ => {
                if in_space {
                    out.push(' ');
                    in_space = false;
                }
                prev_newline_count = 0;
                out.push(ch);
            }
        }
    }

    out
}

// ── Line wrapping ─────────────────────────────────────────────────────────────

/// Greedy word-wrap of `text` to `width` columns.
///
/// Returns `(lines, line_offsets)` where `line_offsets[i]` is the char offset
/// in `text` of the first character of `lines[i]`.
pub fn wrap_text(text: &str, width: usize) -> (Vec<String>, Vec<usize>) {
    let mut lines: Vec<String> = Vec::new();
    let mut offsets: Vec<usize> = Vec::new();

    if width == 0 {
        lines.push(text.to_string());
        offsets.push(0);
        return (lines, offsets);
    }

    // Char offset of the start of the current paragraph within `text`.
    let mut para_start: usize = 0;

    for para in text.split('\n') {
        let para_char_count = para.chars().count();

        if para.trim().is_empty() {
            lines.push(String::new());
            offsets.push(para_start);
            para_start += para_char_count + 1; // +1 for the '\n' separator
            continue;
        }

        // Collect (char_offset_within_para, word) pairs.
        let mut words: Vec<(usize, &str)> = Vec::new();
        let mut in_word = false;
        let mut word_start_byte = 0usize;
        let mut word_start_char = 0usize;
        let mut char_idx = 0usize;

        for (byte_idx, ch) in para.char_indices() {
            if ch.is_whitespace() {
                if in_word {
                    words.push((word_start_char, &para[word_start_byte..byte_idx]));
                    in_word = false;
                }
            } else {
                if !in_word {
                    word_start_byte = byte_idx;
                    word_start_char = char_idx;
                    in_word = true;
                }
            }
            char_idx += 1;
        }
        if in_word {
            words.push((word_start_char, &para[word_start_byte..]));
        }

        // Greedy wrap: build lines, recording the char offset of each line start.
        let mut cur_line = String::new();
        let mut line_start_char: usize = 0; // offset within paragraph

        for (word_char_off, word) in &words {
            let word_len = word.chars().count();
            let needed = if cur_line.is_empty() {
                word_len
            } else {
                cur_line.chars().count() + 1 + word_len
            };

            if !cur_line.is_empty() && needed > width {
                // Emit completed line.
                lines.push(cur_line.clone());
                offsets.push(para_start + line_start_char);
                cur_line.clear();
                line_start_char = *word_char_off;
            }

            if !cur_line.is_empty() {
                cur_line.push(' ');
            }
            cur_line.push_str(word);
        }

        // Emit final (possibly only) line of this paragraph.
        if !cur_line.is_empty() {
            lines.push(cur_line);
            offsets.push(para_start + line_start_char);
        }

        para_start += para_char_count + 1;
    }

    if lines.is_empty() {
        lines.push(String::new());
        offsets.push(0);
    }

    (lines, offsets)
}

// ── Utilities ─────────────────────────────────────────────────────────────────

/// Usable text columns inside the bordered pane for a given terminal width.
fn inner_width(pane_width: u16) -> usize {
    pane_width.saturating_sub(2) as usize // 1 char border each side
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_EPUB_DIR: &str = "MouseAndMotorcycle";

    #[test]
    fn load_epub_and_extract_text() {
        let dir = Path::new(TEST_EPUB_DIR);
        if !dir.exists() {
            eprintln!("Skipping: test book directory not found");
            return;
        }
        let reader = EpubReader::load(dir, 80).expect("load should not error");
        let reader = reader.expect("should find an epub");
        assert!(!reader.full_text.is_empty(), "extracted text should not be empty");
        assert!(reader.full_text.len() > 1000, "should have extracted a reasonable amount of text");
        assert_eq!(reader.lines.len(), reader.line_offsets.len(), "lines/offsets must match");
        // Verify a few lines from the known book
        let full = &reader.full_text;
        assert!(full.contains("Ralph") || full.contains("mouse") || full.contains("motorcycle"),
            "should contain expected book content");
        println!("Extracted {} chars, {} wrapped lines", full.len(), reader.lines.len());
        println!("First 200 chars: {:?}", &full[..full.len().min(200)]);
    }

    #[test]
    fn wrap_text_basic() {
        let text = "The quick brown fox jumps over the lazy dog";
        let (lines, offsets) = wrap_text(text, 20);
        assert_eq!(lines.len(), offsets.len());
        // Every line should fit within 20 chars (allowing for the word that pushed it over)
        for line in &lines {
            // individual words can exceed width; full lines should be reasonable
            assert!(line.len() <= 25, "line too long: {:?}", line);
        }
        // Offsets should be monotonically increasing
        for i in 1..offsets.len() {
            assert!(offsets[i] >= offsets[i - 1], "offsets not monotonic");
        }
        // First offset is always 0
        assert_eq!(offsets[0], 0);
    }

    #[test]
    fn rewrap_restores_position() {
        let dir = Path::new(TEST_EPUB_DIR);
        if !dir.exists() { return; }
        let mut reader = EpubReader::load(dir, 80).unwrap().unwrap();
        // Scroll forward 50 lines
        reader.scroll_by(50);
        let saved_offset = reader.char_offset;
        // Rewrap at a different width
        reader.rewrap(60);
        // char_offset must be unchanged
        assert_eq!(reader.char_offset, saved_offset, "char_offset should survive rewrap");
        // scroll should map to a line whose offset <= saved_offset
        assert!(reader.line_offsets[reader.scroll] <= saved_offset);
        if reader.scroll + 1 < reader.line_offsets.len() {
            assert!(reader.line_offsets[reader.scroll + 1] > saved_offset);
        }
    }
}
