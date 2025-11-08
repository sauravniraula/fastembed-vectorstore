use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::PyType;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

use crate::embedding_model::FastembedEmbeddingModel;

fn get_text_embedder(
    model: EmbeddingModel,
    show_download_progress: Option<bool>,
    cache_directory: Option<PathBuf>,
) -> Option<TextEmbedding> {
    let mut init_options = InitOptions::new(model);
    init_options = init_options.with_show_download_progress(show_download_progress.unwrap_or(true));
    let cd = cache_directory.unwrap_or_else(|| PathBuf::from("fastembed_cache"));
    init_options = init_options.with_cache_dir(cd);

    if let Ok(embedder) = TextEmbedding::try_new(init_options) {
        Some(embedder)
    } else {
        None
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot_product / (norm_a * norm_b)
}

#[pyclass]
pub struct FastembedVectorstore {
    embedder: TextEmbedding,
    embeddings: HashMap<String, Vec<f32>>,
}

#[pymethods]
impl FastembedVectorstore {
    #[new]
    #[pyo3(signature = (model, show_download_progress=None, cache_directory=None))]
    fn new(
        model: &FastembedEmbeddingModel,
        show_download_progress: Option<bool>,
        cache_directory: Option<PathBuf>,
    ) -> PyResult<Self> {
        let embedder = get_text_embedder(
            model.to_embedding_model(),
            show_download_progress,
            cache_directory,
        )
        .expect("Could not initialize TextEmbedding Model");

        Ok(FastembedVectorstore {
            embedder: embedder,
            embeddings: HashMap::new(),
        })
    }

    #[classmethod]
    #[pyo3(signature = (model, path, show_download_progress=None, cache_directory=None))]
    fn load(
        _: &Bound<'_, PyType>,
        model: &FastembedEmbeddingModel,
        path: String,
        show_download_progress: Option<bool>,
        cache_directory: Option<PathBuf>,
    ) -> PyResult<FastembedVectorstore> {
        let file_exists = fs::exists(&path).expect("Could not check if file exists");
        if file_exists {
            let json_string = fs::read_to_string(path).expect("Failed to read file");
            let parsed_json: HashMap<String, Vec<f32>> =
                serde_json::from_str(&json_string).expect("Failed to parse JSON");

            let embedder = get_text_embedder(
                model.to_embedding_model(),
                show_download_progress,
                cache_directory,
            )
            .expect("Could not initialize TextEmbedding Model");

            return Ok(FastembedVectorstore {
                embedder: embedder,
                embeddings: parsed_json,
            });
        } else {
            return Err(PyException::new_err("File doesn't exist"));
        }
    }

    fn embed_documents(&mut self, mut documents: Vec<String>) -> PyResult<bool> {
        if let Ok(mut vec_embeddings) = self.embedder.embed(documents.clone(), None) {
            loop {
                if documents.is_empty() {
                    break;
                }
                if let (Some(popped_document), Some(popped_embedding)) =
                    (documents.pop(), vec_embeddings.pop())
                {
                    self.embeddings.insert(popped_document, popped_embedding);
                }
            }
        }
        Ok(true)
    }

    fn search(&self, query: &str, n: usize) -> PyResult<Vec<(String, f32)>> {
        let query_embeddings = Arc::new(
            self.embedder
                .embed(vec![query.to_string()], None)
                .expect("Could not embed query"),
        );
        let documents_embeddings: Vec<Vec<f32>> = self.embeddings.values().cloned().collect();
        let documents_length = documents_embeddings.len();
        if documents_length == 0 {
            return Ok(Vec::new());
        }
        let mut similarities = Vec::<(usize, f32)>::new();
        let available_threads = std::cmp::max(1, num_cpus::get());
        let chunk_size = std::cmp::max(
            1,
            (documents_length + available_threads - 1) / available_threads,
        );
        let mut index: usize = 0;
        let mut thread_handles = Vec::<JoinHandle<_>>::new();

        loop {
            if index >= documents_length {
                break;
            }
            let query_embeddings_clone = query_embeddings.clone();
            let end_index = std::cmp::min(index + chunk_size, documents_length);
            let chunk_embeddings: Vec<Vec<f32>> = documents_embeddings[index..end_index].to_vec();
            let start_index = index;

            thread_handles.push(thread::spawn(move || {
                let mut chunk_similarities = Vec::<(usize, f32)>::new();
                let mut i = 0;
                loop {
                    if i >= chunk_embeddings.len() {
                        break;
                    }
                    let doc_index = start_index + i;
                    let similarity =
                        cosine_similarity(&query_embeddings_clone[0], &chunk_embeddings[i]);
                    chunk_similarities.push((doc_index, similarity));
                    i += 1;
                }
                chunk_similarities
            }));
            index += chunk_size;
        }

        while let Some(handle) = thread_handles.pop() {
            let chunk_result = handle.join().expect("Thread did not return result");
            similarities.extend(chunk_result);
        }

        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        similarities.truncate(n);

        let documents: Vec<&String> = self.embeddings.keys().collect();
        let result: Vec<(String, f32)> = similarities
            .into_iter()
            .map(|(index, similarity)| (documents[index].clone(), similarity))
            .collect();

        Ok(result)
    }

    fn save(&self, path: &str) -> PyResult<bool> {
        if let Ok(serialized_json) = serde_json::to_string_pretty(&self.embeddings) {
            if let Some(parent) = Path::new(path).parent() {
                if let Err(e) = fs::create_dir_all(parent) {
                    println!("Error creating directory: {}", e);
                    return Ok(false);
                }
            }

            if let Ok(mut save_file) = File::create(path) {
                match save_file.write_all(serialized_json.as_bytes()) {
                    Ok(_) => {}
                    Err(e) => {
                        println!("Error saving file {}", e);
                        return Ok(false);
                    }
                }
            }
        } else {
            println!("Error serializing HashMap");
            return Ok(false);
        }
        Ok(true)
    }
}
