//! TUI Monitor — 3-panel terminal display for real-time inference observability.
//!
//! Panel layout (fixed, no flickering):
//!   TOP    : Speculation bus status (accept rate, SSD read, DMA bandwidth)
//!   CENTER : Layer activation heatmap (block characters, ANSI-256 color)
//!   BOTTOM : Latent direction log stream (structured neuron events)
//!
//! Runs in a dedicated thread at 4 Hz. Does NOT share the GPU inference thread.
//! Crossterm raw mode + alternate screen prevents contamination of stdout.

use crossterm::{
    cursor,
    execute,
    style::{Color, Print, SetForegroundColor, ResetColor},
    terminal::{self, ClearType},
};
use std::collections::VecDeque;
use std::io::{stdout, Write};
use std::sync::mpsc;
use std::time::Duration;

/// Update message sent from the inference thread to the monitor thread.
#[derive(Debug, Clone)]
pub struct MonitorUpdate {
    /// COBER speculative acceptance rate 0.0–1.0.
    pub accept_rate: f32,
    /// Physical SSD read throughput in MB/s (from transport layer).
    pub ssd_speed_mbs: f32,
    /// Effective DMA bandwidth multiplier vs baseline PCIe.
    pub dma_multiplier: f32,
    /// RMS activation norm per transformer layer (empty = not yet computed).
    pub layer_norms: Vec<f32>,
    /// Structured neuron event line to append to the log panel (None = no new line).
    pub log_line: Option<String>,
    /// Current tokens/second measurement.
    pub tps: f32,
}

impl Default for MonitorUpdate {
    fn default() -> Self {
        Self {
            accept_rate: 0.0,
            ssd_speed_mbs: 0.0,
            dma_multiplier: 1.0,
            layer_norms: Vec::new(),
            log_line: None,
            tps: 0.0,
        }
    }
}

/// Handle to the monitor thread. Drop to stop monitoring.
pub struct MonitorHandle {
    tx: mpsc::Sender<MonitorUpdate>,
    _thread: std::thread::JoinHandle<()>,
}

impl MonitorHandle {
    /// Spawn the monitor thread and return a handle for sending updates.
    pub fn spawn() -> Self {
        let (tx, rx) = mpsc::channel::<MonitorUpdate>();
        let thread = std::thread::spawn(move || run_monitor(rx));
        Self { tx, _thread: thread }
    }

    /// Send an update to the monitor thread (non-blocking).
    pub fn update(&self, upd: MonitorUpdate) {
        let _ = self.tx.send(upd);
    }

    /// Log a structured neuron event without changing metrics.
    pub fn log(&self, line: impl Into<String>) {
        let _ = self.tx.send(MonitorUpdate {
            log_line: Some(line.into()),
            ..Default::default()
        });
    }
}

impl Drop for MonitorHandle {
    fn drop(&mut self) {
        // Restore terminal on drop.
        let mut out = stdout();
        let _ = execute!(
            out,
            terminal::LeaveAlternateScreen,
            cursor::Show,
        );
        let _ = terminal::disable_raw_mode();
    }
}

/// Block characters ordered by density (low → high activation).
const BLOCKS: [char; 5] = [' ', '░', '▒', '▓', '█'];

fn density_char(v: f32) -> char {
    let idx = (v.clamp(0.0, 1.0) * 4.0) as usize;
    BLOCKS[idx.min(4)]
}

/// ANSI-256 color: cold (blue) → warm (orange/red) for activation intensity.
fn activation_color(intensity: f32) -> Color {
    let i = intensity.clamp(0.0, 1.0);
    if i < 0.25 {
        Color::AnsiValue(17)  // dark blue
    } else if i < 0.5 {
        Color::AnsiValue(27)  // medium blue
    } else if i < 0.70 {
        Color::AnsiValue(214) // orange
    } else if i < 0.85 {
        Color::AnsiValue(202) // dark orange
    } else {
        Color::AnsiValue(196) // red
    }
}

fn run_monitor(rx: mpsc::Receiver<MonitorUpdate>) {
    let mut out = stdout();

    if execute!(out, terminal::EnterAlternateScreen, cursor::Hide).is_err() {
        return;
    }
    if terminal::enable_raw_mode().is_err() {
        let _ = execute!(out, terminal::LeaveAlternateScreen, cursor::Show);
        return;
    }

    let mut state = MonitorUpdate::default();
    let mut log_lines: VecDeque<String> = VecDeque::with_capacity(20);
    let mut tick: u64 = 0;

    loop {
        // Drain all pending updates without blocking.
        let mut redraw = false;
        while let Ok(upd) = rx.try_recv() {
            if !upd.layer_norms.is_empty() { state.layer_norms = upd.layer_norms; }
            if upd.accept_rate > 0.0 { state.accept_rate = upd.accept_rate; }
            if upd.ssd_speed_mbs > 0.0 { state.ssd_speed_mbs = upd.ssd_speed_mbs; }
            if upd.dma_multiplier != 1.0 { state.dma_multiplier = upd.dma_multiplier; }
            if upd.tps > 0.0 { state.tps = upd.tps; }
            if let Some(line) = upd.log_line {
                log_lines.push_back(line);
                if log_lines.len() > 18 { log_lines.pop_front(); }
            }
            redraw = true;
        }

        // Redraw at 4 Hz even without updates (shows tick counter).
        tick += 1;
        if redraw || tick % 4 == 0 {
            render(&mut out, &state, &log_lines, tick);
        }

        std::thread::sleep(Duration::from_millis(250));
    }
}

fn render(
    out: &mut impl Write,
    state: &MonitorUpdate,
    log_lines: &VecDeque<String>,
    tick: u64,
) {
    let (cols, rows) = terminal::size().unwrap_or((120, 40));
    let w = cols as usize;

    // Clamp panel heights to available space.
    let top_h = 5usize;
    let bot_h = (rows as usize).saturating_sub(top_h + 2).min(20);
    let mid_h = (rows as usize).saturating_sub(top_h + bot_h + 2).max(3);

    let _ = execute!(out, cursor::MoveTo(0, 0));

    // ── TOP PANEL: Speculation Bus ───────────────────────────────────────────
    let sep = "─".repeat(w.saturating_sub(2));
    let _ = execute!(
        out,
        SetForegroundColor(Color::AnsiValue(244)),
        Print(format!("┌{}┐\r\n", sep)),
        ResetColor,
    );

    let accept_pct = state.accept_rate * 100.0;
    let accept_color = if accept_pct >= 70.0 { Color::AnsiValue(40) }
                       else if accept_pct >= 40.0 { Color::AnsiValue(214) }
                       else { Color::AnsiValue(196) };

    let _ = execute!(
        out,
        SetForegroundColor(Color::AnsiValue(244)), Print("│ "),
        SetForegroundColor(Color::AnsiValue(250)), Print("SPECULATION BUS"),
        SetForegroundColor(Color::AnsiValue(244)), Print("  |  "),
        SetForegroundColor(Color::AnsiValue(250)), Print("ALPHA "),
        SetForegroundColor(accept_color),
        Print(format!("{:>5.1}%", accept_pct)),
        SetForegroundColor(Color::AnsiValue(244)), Print("  |  "),
        SetForegroundColor(Color::AnsiValue(250)), Print("SSD "),
        SetForegroundColor(Color::AnsiValue(117)),
        Print(format!("{:>7.1} MB/s", state.ssd_speed_mbs)),
        SetForegroundColor(Color::AnsiValue(244)), Print("  |  "),
        SetForegroundColor(Color::AnsiValue(250)), Print("DMA "),
        SetForegroundColor(Color::AnsiValue(46)),
        Print(format!("{:.2}x", state.dma_multiplier)),
        SetForegroundColor(Color::AnsiValue(244)), Print("  |  "),
        SetForegroundColor(Color::AnsiValue(250)), Print("TPS "),
        SetForegroundColor(Color::AnsiValue(220)),
        Print(format!("{:>6.2}", state.tps)),
        SetForegroundColor(Color::AnsiValue(244)),
        Print(format!("{:>width$}", "│", width = w.saturating_sub(80))),
        Print("\r\n"),
        ResetColor,
    );

    // Tick indicator (heartbeat)
    let heartbeat = ["◆", "◇"][((tick / 2) % 2) as usize];
    let _ = execute!(
        out,
        SetForegroundColor(Color::AnsiValue(244)),
        Print(format!("│ {heartbeat} NodeStor Mechanistic Monitor v1  [Ctrl+C to exit]{:>width$}\r\n",
            "│", width = w.saturating_sub(57))),
        Print(format!("└{}┘\r\n", sep)),
        ResetColor,
    );

    // ── CENTER PANEL: Activation Heatmap ────────────────────────────────────
    let n_layers = state.layer_norms.len().max(1);
    let max_norm = state.layer_norms.iter().cloned().fold(0.0f32, f32::max).max(1e-9);

    // Normalize norms per-row.
    let normalized: Vec<f32> = state.layer_norms.iter()
        .map(|&v| v / max_norm)
        .collect();

    // Each row = one layer, columns = activation blocks across hidden_dim segments.
    let cols_per_layer = w.saturating_sub(6); // leave room for label
    let _ = execute!(
        out,
        SetForegroundColor(Color::AnsiValue(244)),
        Print(format!("┌ ACTIVATION MAP ─ {} layers ─{}┐\r\n",
            n_layers, "─".repeat(w.saturating_sub(27 + n_layers.to_string().len())))),
        ResetColor,
    );

    for (i, &norm) in normalized.iter().enumerate().take(mid_h) {
        let color = activation_color(norm);
        let block = density_char(norm);
        let bar: String = (0..cols_per_layer).map(|col| {
            // Add slight gradient across the layer width
            let x = col as f32 / cols_per_layer as f32;
            let local = (norm * (0.7 + 0.3 * (1.0 - (x - 0.5).abs() * 2.0))).clamp(0.0, 1.0);
            density_char(local)
        }).collect();

        let _ = execute!(
            out,
            SetForegroundColor(Color::AnsiValue(244)),
            Print(format!("│{:>3} ", i)),
            SetForegroundColor(color),
            Print(&bar),
            SetForegroundColor(Color::AnsiValue(244)),
            Print("│\r\n"),
            ResetColor,
        );
    }

    // Fill remaining mid rows if fewer layers than panel height
    for _ in normalized.len()..mid_h {
        let _ = execute!(
            out,
            SetForegroundColor(Color::AnsiValue(240)),
            Print(format!("│    {}{}\r\n", " ".repeat(cols_per_layer), "│")),
            ResetColor,
        );
    }

    let _ = execute!(
        out,
        SetForegroundColor(Color::AnsiValue(244)),
        Print(format!("└{}┘\r\n", "─".repeat(w.saturating_sub(2)))),
        ResetColor,
    );

    // ── BOTTOM PANEL: Latent Direction Log ──────────────────────────────────
    let _ = execute!(
        out,
        SetForegroundColor(Color::AnsiValue(244)),
        Print(format!("┌ LATENT DIRECTION LOG{:─>width$}┐\r\n",
            "─", width = w.saturating_sub(23))),
        ResetColor,
    );

    let visible: Vec<&String> = log_lines.iter()
        .rev()
        .take(bot_h.saturating_sub(2))
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();

    for line in &visible {
        // Truncate to terminal width, pad to fill
        let truncated: String = line.chars().take(w.saturating_sub(4)).collect();
        let _ = execute!(
            out,
            SetForegroundColor(Color::AnsiValue(250)),
            Print(format!("│ {:<width$}│\r\n", truncated, width = w.saturating_sub(4))),
            ResetColor,
        );
    }

    // Pad empty rows
    for _ in visible.len()..bot_h.saturating_sub(2) {
        let _ = execute!(
            out,
            SetForegroundColor(Color::AnsiValue(240)),
            Print(format!("│{:>width$}\r\n", "│", width = w.saturating_sub(1))),
            ResetColor,
        );
    }

    let _ = execute!(
        out,
        SetForegroundColor(Color::AnsiValue(244)),
        Print(format!("└{}┘\r\n", "─".repeat(w.saturating_sub(2)))),
        ResetColor,
    );

    let _ = out.flush();
}
