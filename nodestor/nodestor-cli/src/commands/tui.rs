// NodeStor TUI — clean 3-tab interface: Models | Chat | Connect
// No observability noise. Simple: browse models, chat, connect to other tools.

use std::{
    io::stdout,
    path::PathBuf,
    sync::mpsc,
    time::{Duration, Instant},
};
use anyhow::Result;
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Wrap},
    Frame, Terminal,
};
use ratatui::crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};

const TEAL:  Color = Color::Rgb(0, 200, 180);
const WHITE: Color = Color::White;
const GRAY:  Color = Color::Rgb(120, 120, 130);
const DIM:   Color = Color::Rgb(50, 50, 65);
const BG:    Color = Color::Rgb(8, 8, 14);
const AMBER: Color = Color::Rgb(240, 180, 50);
const GREEN: Color = Color::Rgb(60, 220, 100);

#[derive(Clone, Copy, PartialEq)]
enum Tab { Models, Chat, Connect }

struct ModelEntry {
    path:    PathBuf,
    name:    String,
    size_gb: f32,
}

#[derive(Clone)]
enum ChatMsg {
    User(String),
    Assistant(String),
    Thinking,
}

enum BgEvent {
    ChatResponse(String),
    ChatError(String),
    DownloadProgress(f32, u64),
    DownloadDone,
    DownloadError(String),
}

#[derive(PartialEq)]
enum InputMode { Normal, Chat, DownloadRepo, DownloadFile }

struct App {
    tab:        Tab,
    models:     Vec<ModelEntry>,
    model_sel:  ListState,
    chat:       Vec<ChatMsg>,
    input:      String,
    mode:       InputMode,
    dl_repo:    String,
    server_url: String,
    status:     String,
    status_ts:  Instant,
    tx:         mpsc::Sender<BgEvent>,
    rx:         mpsc::Receiver<BgEvent>,
}

impl App {
    fn new() -> Self {
        let (tx, rx) = mpsc::channel();
        let mut a = Self {
            tab:        Tab::Models,
            models:     Vec::new(),
            model_sel:  ListState::default(),
            chat:       Vec::new(),
            input:      String::new(),
            mode:       InputMode::Normal,
            dl_repo:    String::new(),
            server_url: "http://localhost:8080".into(),
            status:     "q quit  Tab switch tab".into(),
            status_ts:  Instant::now(),
            tx, rx,
        };
        a.scan();
        if !a.models.is_empty() { a.model_sel.select(Some(0)); }
        a
    }

    fn scan(&mut self) {
        self.models.clear();
        let mut dirs = Vec::new();
        if let Some(mut h) = dirs::home_dir() {
            h.push(".nodestor"); h.push("models");
            dirs.push(h);
        }
        dirs.push(PathBuf::from("."));
        for dir in &dirs {
            if let Ok(rd) = std::fs::read_dir(dir) {
                for e in rd.flatten() {
                    let p = e.path();
                    if p.extension().map(|x| x == "gguf").unwrap_or(false) {
                        let size = e.metadata().map(|m| m.len()).unwrap_or(0);
                        let name = p.file_name().unwrap_or_default().to_string_lossy().into();
                        self.models.push(ModelEntry { path: p, name, size_gb: size as f32 / 1e9 });
                    }
                }
            }
        }
        self.models.sort_by(|a, b| b.size_gb.partial_cmp(&a.size_gb)
            .unwrap_or(std::cmp::Ordering::Equal));
        let n = self.models.len();
        if n == 0 {
            self.model_sel.select(None);
        } else if self.model_sel.selected().map(|i| i >= n).unwrap_or(true) {
            self.model_sel.select(Some(0));
        }
    }

    fn set_status(&mut self, msg: impl Into<String>) {
        self.status = msg.into();
        self.status_ts = Instant::now();
    }

    fn poll_bg(&mut self) {
        while let Ok(ev) = self.rx.try_recv() {
            match ev {
                BgEvent::ChatResponse(text) => {
                    match self.chat.last_mut() {
                        Some(ChatMsg::Thinking) => {
                            *self.chat.last_mut().unwrap() = ChatMsg::Assistant(text);
                        }
                        Some(ChatMsg::Assistant(s)) => s.push_str(&text),
                        _ => self.chat.push(ChatMsg::Assistant(text)),
                    }
                    self.set_status("Done");
                }
                BgEvent::ChatError(e) => {
                    if matches!(self.chat.last(), Some(ChatMsg::Thinking)) {
                        *self.chat.last_mut().unwrap() =
                            ChatMsg::Assistant(format!("[Error: {}]", e));
                    }
                    self.set_status("Error — run: nodestor serve --model <file.gguf>");
                }
                BgEvent::DownloadProgress(pct, bytes) => {
                    self.set_status(format!("Downloading {:.0}%  {:.1} MB", pct * 100.0, bytes as f32 / 1e6));
                }
                BgEvent::DownloadDone => {
                    self.set_status("Download complete — rescanning...");
                    self.scan();
                }
                BgEvent::DownloadError(e) => {
                    self.set_status(format!("Download error: {}", e));
                }
            }
        }
    }

    fn send_chat(&mut self) {
        let msg = std::mem::take(&mut self.input);
        if msg.trim().is_empty() { return; }
        let history: Vec<serde_json::Value> = self.chat.iter().filter_map(|m| match m {
            ChatMsg::User(s) => Some(serde_json::json!({"role":"user","content":s})),
            ChatMsg::Assistant(s) if !s.starts_with("[Error") =>
                Some(serde_json::json!({"role":"assistant","content":s})),
            _ => None,
        }).collect();
        self.chat.push(ChatMsg::User(msg));
        self.chat.push(ChatMsg::Thinking);
        self.mode = InputMode::Normal;
        let url = format!("{}/v1/chat/completions", self.server_url);
        let tx = self.tx.clone();
        std::thread::spawn(move || {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let body = serde_json::json!({
                    "model": "nodestor",
                    "messages": history,
                    "stream": false,
                    "max_tokens": 512,
                });
                match reqwest::Client::new().post(&url).json(&body).send().await {
                    Err(e) => { let _ = tx.send(BgEvent::ChatError(e.to_string())); }
                    Ok(r)  => match r.json::<serde_json::Value>().await {
                        Err(e) => { let _ = tx.send(BgEvent::ChatError(e.to_string())); }
                        Ok(v)  => {
                            let t = v["choices"][0]["message"]["content"]
                                .as_str().unwrap_or("").to_string();
                            let _ = tx.send(BgEvent::ChatResponse(t));
                        }
                    }
                }
            });
        });
    }

    fn start_download(&mut self) {
        let repo = self.dl_repo.clone();
        let file = std::mem::take(&mut self.input);
        self.mode = InputMode::Normal;
        if repo.trim().is_empty() || file.trim().is_empty() { return; }
        let mut dest = dirs::home_dir().unwrap_or_default();
        dest.push(".nodestor"); dest.push("models");
        let tx = self.tx.clone();
        self.set_status(format!("Starting download: {}...", file));
        std::thread::spawn(move || {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let url = format!("https://huggingface.co/{}/resolve/main/{}", repo, file);
                let _ = tokio::fs::create_dir_all(&dest).await;
                let res = match reqwest::Client::new().get(&url).send().await {
                    Err(e) => { let _ = tx.send(BgEvent::DownloadError(e.to_string())); return; }
                    Ok(r)  => r,
                };
                if !res.status().is_success() {
                    let _ = tx.send(BgEvent::DownloadError(format!("HTTP {}", res.status())));
                    return;
                }
                let total = res.content_length().unwrap_or(1);
                let mut f = match tokio::fs::File::create(dest.join(&file)).await {
                    Err(e) => { let _ = tx.send(BgEvent::DownloadError(e.to_string())); return; }
                    Ok(f)  => f,
                };
                let mut got = 0u64;
                let mut stream = res.bytes_stream();
                use futures::StreamExt;
                use tokio::io::AsyncWriteExt;
                while let Some(chunk) = stream.next().await {
                    match chunk {
                        Err(e) => { let _ = tx.send(BgEvent::DownloadError(e.to_string())); return; }
                        Ok(d)  => {
                            let _ = f.write_all(&d).await;
                            got += d.len() as u64;
                            let _ = tx.send(BgEvent::DownloadProgress(
                                got as f32 / total as f32, got,
                            ));
                        }
                    }
                }
                let _ = tx.send(BgEvent::DownloadDone);
            });
        });
    }
}

// ── Drawing ────────────────────────────────────────────────────────────────────

fn draw(f: &mut Frame, app: &mut App) {
    let area = f.area();
    f.render_widget(Block::default().style(Style::default().bg(BG)), area);

    let chunks = Layout::vertical([
        Constraint::Length(3),
        Constraint::Fill(1),
        Constraint::Length(1),
    ]).split(area);

    draw_header(f, chunks[0], app);
    match app.tab {
        Tab::Models  => draw_models(f, chunks[1], app),
        Tab::Chat    => draw_chat(f, chunks[1], app),
        Tab::Connect => draw_connect(f, chunks[1], app),
    }
    draw_footer(f, chunks[2], app);
}

fn tab_style(active: bool) -> Style {
    if active {
        Style::default().fg(Color::Black).bg(TEAL).add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(GRAY)
    }
}

fn draw_header(f: &mut Frame, area: Rect, app: &App) {
    let spans = vec![
        Span::styled("  NodeStor  ", Style::default().fg(TEAL).add_modifier(Modifier::BOLD)),
        Span::styled(" 1 ", tab_style(app.tab == Tab::Models)),
        Span::styled(" Models ", tab_style(app.tab == Tab::Models)),
        Span::styled("  2 ", tab_style(app.tab == Tab::Chat)),
        Span::styled(" Chat ", tab_style(app.tab == Tab::Chat)),
        Span::styled("  3 ", tab_style(app.tab == Tab::Connect)),
        Span::styled(" Connect ", tab_style(app.tab == Tab::Connect)),
        Span::styled(format!("   —  {}", app.server_url), Style::default().fg(DIM)),
    ];
    f.render_widget(
        Paragraph::new(Line::from(spans))
            .block(Block::default().borders(Borders::BOTTOM).border_style(Style::default().fg(DIM))),
        area,
    );
}

fn draw_models(f: &mut Frame, area: Rect, app: &mut App) {
    let chunks = Layout::horizontal([
        Constraint::Fill(1),
        Constraint::Length(34),
    ]).split(area);

    let items: Vec<ListItem> = if app.models.is_empty() {
        vec![
            ListItem::new(Line::from(Span::styled("  No models found.", Style::default().fg(GRAY)))),
            ListItem::new(Line::from(Span::styled("  Press D to download.", Style::default().fg(DIM)))),
        ]
    } else {
        app.models.iter().map(|m| {
            ListItem::new(Line::from(vec![
                Span::styled(format!(" {:<44}", truncate(&m.name, 44)), Style::default().fg(WHITE)),
                Span::styled(format!("{:>6.1} GB ", m.size_gb), Style::default().fg(GRAY)),
            ]))
        }).collect()
    };

    let list = List::new(items)
        .block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(DIM))
            .title(Span::styled(" Models ", Style::default().fg(TEAL))))
        .highlight_style(Style::default().bg(Color::Rgb(12, 40, 50)).fg(TEAL).add_modifier(Modifier::BOLD))
        .highlight_symbol("▶ ");
    f.render_stateful_widget(list, chunks[0], &mut app.model_sel);

    let info = match app.model_sel.selected().and_then(|i| app.models.get(i)) {
        None => vec![
            Line::from(""),
            Line::from(Span::styled(" No model.", Style::default().fg(GRAY))),
        ],
        Some(m) => vec![
            Line::from(""),
            Line::from(Span::styled(format!(" {}", truncate(&m.name, 28)),
                Style::default().fg(WHITE).add_modifier(Modifier::BOLD))),
            Line::from(""),
            Line::from(Span::styled(format!(" {:.2} GB", m.size_gb), Style::default().fg(GRAY))),
            Line::from(""),
            Line::from(Span::styled(" nodestor serve \\", Style::default().fg(TEAL))),
            Line::from(Span::styled(format!("  --model {}", truncate(&m.name, 22)),
                Style::default().fg(TEAL))),
        ],
    };
    f.render_widget(
        Paragraph::new(info)
            .block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(DIM))
                .title(Span::styled(" Info ", Style::default().fg(GRAY)))),
        chunks[1],
    );

    // Download dialog overlay
    if app.mode == InputMode::DownloadRepo || app.mode == InputMode::DownloadFile {
        let popup = centered_rect(64, 8, area);
        f.render_widget(Block::default().style(Style::default().bg(BG)), popup);
        let (title, prompt, hint) = match app.mode {
            InputMode::DownloadRepo => (
                " Download — Step 1: HuggingFace Repo ",
                "Repo ID (e.g. TheBloke/Mistral-7B-GGUF):",
                "Enter confirm  Esc cancel",
            ),
            _ => (
                " Download — Step 2: File name ",
                "Filename (e.g. mistral-7b.Q4_K_M.gguf):",
                "Enter start download  Esc cancel",
            ),
        };
        f.render_widget(
            Paragraph::new(vec![
                Line::from(""),
                Line::from(Span::styled(format!(" {}", prompt), Style::default().fg(GRAY))),
                Line::from(Span::styled(
                    format!("  > {}_", app.input),
                    Style::default().fg(WHITE).add_modifier(Modifier::BOLD),
                )),
                Line::from(""),
                Line::from(Span::styled(format!("  {}", hint), Style::default().fg(DIM))),
            ])
            .block(Block::default().borders(Borders::ALL)
                .border_style(Style::default().fg(TEAL))
                .title(Span::styled(title, Style::default().fg(TEAL)))),
            popup,
        );
    }
}

fn draw_chat(f: &mut Frame, area: Rect, app: &App) {
    let chunks = Layout::vertical([
        Constraint::Fill(1),
        Constraint::Length(3),
    ]).split(area);

    let msgs: Vec<Line> = app.chat.iter().flat_map(|m| match m {
        ChatMsg::User(s) => vec![
            Line::from(Span::styled("You:", Style::default().fg(TEAL).add_modifier(Modifier::BOLD))),
            Line::from(Span::styled(format!("  {}", s), Style::default().fg(WHITE))),
            Line::from(""),
        ],
        ChatMsg::Assistant(s) => vec![
            Line::from(Span::styled("NodeStor:", Style::default().fg(AMBER).add_modifier(Modifier::BOLD))),
            Line::from(Span::styled(format!("  {}", s), Style::default().fg(WHITE))),
            Line::from(""),
        ],
        ChatMsg::Thinking => vec![
            Line::from(Span::styled("NodeStor: ▌", Style::default().fg(AMBER))),
        ],
    }).collect();

    let scroll = msgs.len().saturating_sub(chunks[0].height.saturating_sub(2) as usize) as u16;
    f.render_widget(
        Paragraph::new(msgs)
            .block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(DIM))
                .title(Span::styled(
                    " Chat  (requires: nodestor serve --model <file.gguf>) ",
                    Style::default().fg(TEAL),
                )))
            .wrap(Wrap { trim: false })
            .scroll((scroll, 0)),
        chunks[0],
    );

    let (border_c, content) = if app.mode == InputMode::Chat {
        (TEAL, format!(" > {}_", app.input))
    } else {
        (DIM, " Press i to type a message...".into())
    };
    f.render_widget(
        Paragraph::new(content)
            .style(Style::default().fg(if app.mode == InputMode::Chat { WHITE } else { GRAY }))
            .block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(border_c))),
        chunks[1],
    );
}

fn draw_connect(f: &mut Frame, area: Rect, app: &App) {
    let chunks = Layout::vertical([
        Constraint::Length(8),
        Constraint::Length(11),
        Constraint::Fill(1),
    ]).split(area);

    let url = &app.server_url;
    f.render_widget(
        Paragraph::new(vec![
            Line::from(""),
            Line::from(vec![
                Span::styled("  OpenAI-compatible  ", Style::default().fg(GRAY)),
                Span::styled(url.as_str(), Style::default().fg(TEAL).add_modifier(Modifier::BOLD)),
            ]),
            Line::from(Span::styled("  POST /v1/chat/completions  POST /v1/completions  GET /v1/models", Style::default().fg(WHITE))),
            Line::from(""),
            Line::from(vec![
                Span::styled("  Ollama-compatible   ", Style::default().fg(GRAY)),
                Span::styled(url.as_str(), Style::default().fg(TEAL).add_modifier(Modifier::BOLD)),
            ]),
            Line::from(Span::styled("  POST /api/generate  POST /api/chat  GET /api/tags  GET /api/version", Style::default().fg(WHITE))),
        ])
        .block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(DIM))
            .title(Span::styled(" API Endpoints ", Style::default().fg(TEAL)))),
        chunks[0],
    );

    f.render_widget(
        Paragraph::new(vec![
            Line::from(""),
            Line::from(vec![Span::styled("  LM Studio    ", Style::default().fg(TEAL).add_modifier(Modifier::BOLD)), Span::styled("→ OpenAI Base URL: ", Style::default().fg(GRAY)), Span::styled(url.as_str(), Style::default().fg(WHITE))]),
            Line::from(""),
            Line::from(vec![Span::styled("  OpenWebUI    ", Style::default().fg(TEAL).add_modifier(Modifier::BOLD)), Span::styled("→ Connections → Ollama: ", Style::default().fg(GRAY)), Span::styled(url.as_str(), Style::default().fg(WHITE))]),
            Line::from(""),
            Line::from(vec![Span::styled("  Cursor       ", Style::default().fg(TEAL).add_modifier(Modifier::BOLD)), Span::styled("→ Base URL: ", Style::default().fg(GRAY)), Span::styled(format!("{}/v1", url), Style::default().fg(WHITE))]),
            Line::from(""),
            Line::from(vec![Span::styled("  Claude Code  ", Style::default().fg(TEAL).add_modifier(Modifier::BOLD)), Span::styled("→ ANTHROPIC_BASE_URL=", Style::default().fg(GRAY)), Span::styled(format!("{}/v1", url), Style::default().fg(WHITE))]),
            Line::from(""),
            Line::from(vec![Span::styled("  Any OpenAI SDK ", Style::default().fg(TEAL).add_modifier(Modifier::BOLD)), Span::styled(format!("base_url=\"{}/v1\"  api_key=\"any\"", url), Style::default().fg(WHITE))]),
        ])
        .block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(DIM))
            .title(Span::styled(" Connect from any tool ", Style::default().fg(TEAL)))),
        chunks[1],
    );

    f.render_widget(
        Paragraph::new(vec![
            Line::from(""),
            Line::from(Span::styled("  # Install NodeStor", Style::default().fg(GRAY))),
            Line::from(Span::styled("  cargo install nodestor", Style::default().fg(GREEN))),
            Line::from(""),
            Line::from(Span::styled("  # Download a model from HuggingFace", Style::default().fg(GRAY))),
            Line::from(Span::styled("  nodestor pull TheBloke/Mistral-7B-GGUF --filename mistral-7b.Q4_K_M.gguf", Style::default().fg(GREEN))),
            Line::from(""),
            Line::from(Span::styled("  # Start the server", Style::default().fg(GRAY))),
            Line::from(Span::styled("  nodestor serve --model ~/.nodestor/models/mistral-7b.Q4_K_M.gguf --port 8080", Style::default().fg(GREEN))),
            Line::from(""),
            Line::from(Span::styled("  # List local models", Style::default().fg(GRAY))),
            Line::from(Span::styled("  nodestor ls", Style::default().fg(GREEN))),
        ])
        .block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(DIM))
            .title(Span::styled(" Quick Start ", Style::default().fg(GRAY)))),
        chunks[2],
    );
}

fn draw_footer(f: &mut Frame, area: Rect, app: &App) {
    let status = if app.status_ts.elapsed() < Duration::from_secs(8) {
        format!("  {}  │", app.status)
    } else {
        String::new()
    };
    let hints = match (&app.tab, &app.mode) {
        (_, InputMode::Chat)         => "  Enter send  Esc cancel",
        (_, InputMode::DownloadRepo) => "  Enter confirm repo  Esc cancel",
        (_, InputMode::DownloadFile) => "  Enter start download  Esc cancel",
        (Tab::Models, _) => "  ↑↓ navigate  D download  R rescan  q quit  Tab switch",
        (Tab::Chat, _)   => "  i type  q quit  Tab switch",
        (Tab::Connect, _) => "  q quit  Tab switch",
    };
    f.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(status, Style::default().fg(AMBER)),
            Span::styled(hints, Style::default().fg(GRAY)),
            Span::styled("  1/2/3 tabs", Style::default().fg(DIM)),
        ])).style(Style::default().bg(Color::Rgb(10, 10, 18))),
        area,
    );
}

fn centered_rect(w: u16, h: u16, r: Rect) -> Rect {
    let x = r.x + r.width.saturating_sub(w) / 2;
    let y = r.y + r.height.saturating_sub(h) / 2;
    Rect::new(x, y, w.min(r.width), h.min(r.height))
}

fn truncate(s: &str, n: usize) -> String {
    if s.len() <= n { s.to_string() }
    else { format!("{}…", &s[..n.saturating_sub(1)]) }
}

// ── Entry point ────────────────────────────────────────────────────────────────

pub async fn cmd_tui() -> Result<()> {
    enable_raw_mode()?;
    execute!(stdout(), EnterAlternateScreen)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(stdout()))?;
    let mut app = App::new();

    loop {
        app.poll_bg();
        terminal.draw(|f| draw(f, &mut app))?;

        if !event::poll(Duration::from_millis(60))? { continue; }
        let Event::Key(k) = event::read()? else { continue };
        if k.kind != KeyEventKind::Press { continue; }

        match &app.mode {
            InputMode::Chat => match k.code {
                KeyCode::Esc       => { app.input.clear(); app.mode = InputMode::Normal; }
                KeyCode::Enter     => app.send_chat(),
                KeyCode::Backspace => { app.input.pop(); }
                KeyCode::Char(c)   => app.input.push(c),
                _ => {}
            },
            InputMode::DownloadRepo => match k.code {
                KeyCode::Esc     => { app.input.clear(); app.mode = InputMode::Normal; }
                KeyCode::Enter   => {
                    app.dl_repo = std::mem::take(&mut app.input);
                    app.mode = InputMode::DownloadFile;
                }
                KeyCode::Backspace => { app.input.pop(); }
                KeyCode::Char(c) => app.input.push(c),
                _ => {}
            },
            InputMode::DownloadFile => match k.code {
                KeyCode::Esc       => { app.input.clear(); app.mode = InputMode::Normal; }
                KeyCode::Enter     => app.start_download(),
                KeyCode::Backspace => { app.input.pop(); }
                KeyCode::Char(c)   => app.input.push(c),
                _ => {}
            },
            InputMode::Normal => match k.code {
                KeyCode::Char('q') | KeyCode::Char('Q') => break,
                KeyCode::Char('1') => app.tab = Tab::Models,
                KeyCode::Char('2') => app.tab = Tab::Chat,
                KeyCode::Char('3') => app.tab = Tab::Connect,
                KeyCode::Tab => {
                    app.tab = match app.tab {
                        Tab::Models  => Tab::Chat,
                        Tab::Chat    => Tab::Connect,
                        Tab::Connect => Tab::Models,
                    };
                }
                KeyCode::Up if app.tab == Tab::Models => {
                    let i = app.model_sel.selected().unwrap_or(0);
                    if i > 0 { app.model_sel.select(Some(i - 1)); }
                }
                KeyCode::Down if app.tab == Tab::Models => {
                    let i = app.model_sel.selected().unwrap_or(0);
                    if i + 1 < app.models.len() { app.model_sel.select(Some(i + 1)); }
                }
                KeyCode::Char('d') | KeyCode::Char('D') if app.tab == Tab::Models => {
                    app.input.clear();
                    app.mode = InputMode::DownloadRepo;
                }
                KeyCode::Char('r') | KeyCode::Char('R') if app.tab == Tab::Models => {
                    app.scan();
                    app.set_status("Rescanned.");
                }
                KeyCode::Char('i') if app.tab == Tab::Chat => {
                    app.mode = InputMode::Chat;
                }
                _ => {}
            },
        }
    }

    disable_raw_mode()?;
    execute!(stdout(), LeaveAlternateScreen)?;
    Ok(())
}
