use pyo3::prelude::*;
use std::sync::Arc;
use nodestor_core::get_metrics_text;
use nodestor_scanner::scan as rust_scan;
use nodestor_inference::pipeline::{InferenceConfig, InferencePipeline};

#[pyclass]
struct NodeStorEngine {
    inner: Arc<InferencePipeline>,
    rt: tokio::runtime::Runtime,
}

#[pyclass]
struct NodeStorBuffer {
    data: Vec<u8>, // Em um caso real, isso seria um mapeamento de memória
}


#[pymethods]
impl NodeStorEngine {
    #[new]
    fn new(model_path: String) -> PyResult<Self> {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let config = InferenceConfig {
            model_path: model_path.clone(),
            prefetch_depth: 4,
            buffer_size: 64 * 1024 * 1024,
        };

        let pipeline = rust_scan()
            .and_then(|_| InferencePipeline::init(config))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(NodeStorEngine {
            inner: Arc::new(pipeline),
            rt,
        })
    }

    fn generate(&self, prompt: String, max_tokens: usize) -> PyResult<String> {
        let pipeline = self.inner.clone();
        let (text, _stats) = self.rt.block_on(async move {
            pipeline.generate(&prompt, max_tokens, None).await
        }).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        Ok(text)
    }

    fn scan(&self) -> PyResult<String> {
        let profile = rust_scan()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(format!("{:?}", profile))
    }

    fn search(&self, query_vector: Vec<f32>, k: usize) -> PyResult<String> {
        Ok(format!("Busca vetorial simulada k={}", k))
    }

    fn get_buffer(&self) -> PyResult<NodeStorBuffer> {
        // Simulação de acesso a buffer de pesos
        Ok(NodeStorBuffer { data: vec![0u8; 1024] })
    }

    fn get_metrics(&self) -> PyResult<String> {
        Ok(get_metrics_text())
    }
}

#[pyfunction]
fn scan() -> PyResult<String> {
    let profile = rust_scan()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(format!("{:?}", profile))
}

/// Módulo de extensão nativo. Exposto ao Python como `nodestor._nodestor`;
/// o pacote Python `nodestor` (CLI + UX) o envolve.
#[pymodule]
fn _nodestor(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(scan, m)?)?;
    m.add_class::<NodeStorEngine>()?;
    Ok(())
}
