use prometheus::{Encoder, TextEncoder, Registry, Counter, Gauge, histogram_opts, Histogram};
use std::sync::LazyLock;

pub static REGISTRY: LazyLock<Registry> = LazyLock::new(|| Registry::new());

pub static IO_THROUGHPUT_BPS: LazyLock<Gauge> = LazyLock::new(|| {
    let g = Gauge::new("nodestor_io_throughput_bps", "Current IO throughput in bytes per second").unwrap();
    REGISTRY.register(Box::new(g.clone())).unwrap();
    g
});

pub static VRAM_USAGE_BYTES: LazyLock<Gauge> = LazyLock::new(|| {
    let g = Gauge::new("nodestor_vram_usage_bytes", "Current VRAM usage in bytes").unwrap();
    REGISTRY.register(Box::new(g.clone())).unwrap();
    g
});

pub static TTFT_MS: LazyLock<Histogram> = LazyLock::new(|| {
    let opts = histogram_opts!("nodestor_ttft_ms", "Time To First Token in milliseconds");
    let h = Histogram::with_opts(opts).unwrap();
    REGISTRY.register(Box::new(h.clone())).unwrap();
    h
});

pub static INFERENCE_TOKENS_TOTAL: LazyLock<Counter> = LazyLock::new(|| {
    let c = Counter::new("nodestor_inference_tokens_total", "Total number of tokens generated").unwrap();
    REGISTRY.register(Box::new(c.clone())).unwrap();
    c
});

pub fn get_metrics_text() -> String {
    let mut buffer = Vec::new();
    let encoder = TextEncoder::new();
    let metric_families = REGISTRY.gather();
    encoder.encode(&metric_families, &mut buffer).unwrap();
    String::from_utf8(buffer).unwrap()
}
