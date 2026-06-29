use std::collections::HashSet;

#[derive(Debug, Clone, Default)]
pub struct IndexerConfig {
    pub batch_size: usize,
    pub chunk_size: usize,
}

#[derive(Debug, Clone)]
pub struct IndexJob {
    pub path: String,
    pub text: String,
    pub urgent: bool,
}

impl IndexJob {
    pub fn new_text(path: &str, text: &str) -> Self {
        Self { path: path.to_string(), text: text.to_string(), urgent: false }
    }
    pub fn new_urgent(path: &str, text: &str) -> Self {
        Self { path: path.to_string(), text: text.to_string(), urgent: true }
    }
}

#[derive(Debug, Default)]
pub struct IndexerStats {
    pub jobs_completed: usize,
    pub jobs_failed: usize,
    pub chunks_indexed: usize,
}

impl IndexerStats {
    pub fn success_rate(&self) -> f32 {
        let total = self.jobs_completed + self.jobs_failed;
        if total == 0 { 1.0 } else { self.jobs_completed as f32 / total as f32 }
    }
}

#[derive(Debug)]
pub struct BatchResult {
    pub jobs_completed: usize,
    pub chunks_total: usize,
}

pub struct BackgroundIndexer {
    config: IndexerConfig,
    queue: Vec<IndexJob>,
    indexed_paths: HashSet<String>,
    paused: bool,
    pub stats: IndexerStats,
}

impl BackgroundIndexer {
    pub fn new(config: IndexerConfig) -> Self {
        Self {
            config,
            queue: Vec::new(),
            indexed_paths: HashSet::new(),
            paused: false,
            stats: IndexerStats::default(),
        }
    }

    pub fn enqueue(&mut self, job: IndexJob) -> Result<(), String> {
        // Deduplication: silently ignore same path
        if !self.indexed_paths.contains(&job.path) {
            self.queue.push(job);
        }
        Ok(())
    }

    pub fn queue_len(&self) -> usize {
        self.queue.len()
    }

    pub fn process_batch(&mut self, max_jobs: usize) -> BatchResult {
        if self.paused {
            return BatchResult { jobs_completed: 0, chunks_total: 0 };
        }
        let n = max_jobs.min(self.queue.len());
        let mut jobs_completed = 0;
        let mut chunks_total = 0;
        let jobs: Vec<IndexJob> = self.queue.drain(..n).collect();
        for job in jobs {
            let chunks = (job.text.len() / 512).max(1);
            self.indexed_paths.insert(job.path);
            self.stats.jobs_completed += 1;
            self.stats.chunks_indexed += chunks;
            jobs_completed += 1;
            chunks_total += chunks;
        }
        BatchResult { jobs_completed, chunks_total }
    }

    pub fn pause(&mut self) { self.paused = true; }
    pub fn resume(&mut self) { self.paused = false; }
}
