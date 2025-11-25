use pyo3::prelude::*;

pub mod embedding_model;
pub mod vector_store;

use crate::embedding_model::FastembedEmbeddingModel;
use crate::vector_store::FastembedVectorstore;

#[pymodule]
fn fastembed_vectorstore(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<FastembedVectorstore>()?;
    m.add_class::<FastembedEmbeddingModel>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search() {
        let mut vectorstore =
            FastembedVectorstore::new(&FastembedEmbeddingModel::BGESmallENV15, None, None)
                .expect("Could not create vectorstore");
        vectorstore
            .embed_documents(vec![
                String::from("The quick brown fox jumps over the lazy dog"),
                String::from("A quick brown dog jumps over the lazy fox"),
                String::from("The lazy fox sleeps while the quick brown dog watches"),
                String::from("Python is a programming language"),
                String::from("Rust is a systems programming language"),
            ])
            .expect("Could not embed documents");

        let results = vectorstore
            .search("What is Python?", 1)
            .expect("Could not search");
        assert_eq!(results.len(), 1);
        assert!(results[0].0 == "Python is a programming language");
    }
}
