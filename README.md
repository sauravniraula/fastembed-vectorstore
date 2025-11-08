# FastEmbed VectorStore

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?style=flat-square&logo=github)](https://github.com/sauravniraula/fastembed_vectorstore)

A high-performance, Rust-based in-memory vector store with FastEmbed integration for Python applications.

## Overview

FastEmbed VectorStore is a lightweight, fast vector database that leverages the power of Rust and the [FastEmbed library](https://github.com/Anush008/fastembed-rs) to provide efficient text embedding and similarity search capabilities. It's designed for applications that need quick semantic search without the overhead of external database systems.

## Features

- 🚀 **High Performance**: Built in Rust with Python bindings for optimal speed
- 🧠 **Multiple Embedding Models**: Support for 30+ pre-trained embedding models including BGE, Nomic, GTE, and more
- 💾 **In-Memory Storage**: Fast in-memory vector storage with persistence capabilities
- 🔍 **Similarity Search**: Cosine similarity-based search with customizable result limits
- 💾 **Save/Load**: Persist and restore vector stores to/from JSON files
- 🐍 **Python Integration**: Seamless Python API with PyO3 bindings

## Supported Embedding Models

The library supports a wide variety of embedding models:

- **BGE Models**: BGEBaseENV15, BGELargeENV15, BGESmallENV15 (with quantized variants)
- **Nomic Models**: NomicEmbedTextV1, NomicEmbedTextV15 (with quantized variants)
- **GTE Models**: GTEBaseENV15, GTELargeENV15 (with quantized variants)
- **Multilingual Models**: MultilingualE5Small, MultilingualE5Base, MultilingualE5Large
- **Specialized Models**: ClipVitB32, JinaEmbeddingsV2BaseCode, ModernBertEmbedLarge
- **And many more...**

## Installation

### Prerequisites

- Python 3.8 or higher
- Rust toolchain (to build from source)


### Install from PyPI

```bash
pip install fastembed-vectorstore
```

### From Source

1. Clone the repository:
```bash
git clone https://github.com/sauravniraula/fastembed_vectorstore.git
cd fastembed_vectorstore
```

2. Install the package:
```bash
maturin develop
```

## Quick Start

```python
from fastembed_vectorstore import FastembedVectorstore, FastembedEmbeddingModel

# Initialize with a model
model = FastembedEmbeddingModel.BGESmallENV15
vectorstore = FastembedVectorstore(model)

# Optionally, customize initialization
# vectorstore = FastembedVectorstore(
#     model,
#     show_download_progress=False,           # default: True
#     cache_directory="./fastembed",          # default: ./fastembed
# )

# Add documents
documents = [
    "The quick brown fox jumps over the lazy dog",
    "A quick brown dog jumps over the lazy fox",
    "The lazy fox sleeps while the quick brown dog watches",
    "Python is a programming language",
    "Rust is a systems programming language"
]

# Embed and store documents
success = vectorstore.embed_documents(documents)
print(f"Documents embedded: {success}")

# Search for similar documents
query = "What is Python?"
results = vectorstore.search(query, n=3)

for doc, similarity in results:
    print(f"Document: {doc}")
    print(f"Similarity: {similarity:.4f}")
    print("---")

# Save the vector store
vectorstore.save("my_vectorstore.json")

# Load the vector store later
loaded_vectorstore = FastembedVectorstore.load(model, "my_vectorstore.json")
```

## API Reference

### FastembedEmbeddingModel

Enum containing all supported embedding models. Choose based on your use case:

- **Small models**: Faster, lower memory usage (e.g., `BGESmallENV15`)
- **Base models**: Balanced performance (e.g., `BGEBaseENV15`)
- **Large models**: Higher quality embeddings (e.g., `BGELargeENV15`)
- **Quantized models**: Reduced memory usage (e.g., `BGESmallENV15Q`)

### FastembedVectorstore

#### Constructor
```python
vectorstore = FastembedVectorstore(
    model: FastembedEmbeddingModel,
    show_download_progress: bool | None = ...,
    cache_directory: str | os.PathLike[str] | None = ...,
)
```
Args:
- `model`: Embedding model to use.
- `show_download_progress`: Whether to show model download progress. Defaults to True.
- `cache_directory`: Directory to cache/download model files. Defaults to `./fastembed`.

#### Methods

##### `embed_documents(documents: List[str]) -> bool`
Embeds a list of documents and stores them in the vector store.

##### `search(query: str, n: int) -> List[Tuple[str, float]]`
Searches for the most similar documents to the query. Returns a list of tuples containing (document, similarity_score).

##### `save(path: str) -> bool`
Saves the vector store to a JSON file.

##### `load(model: FastembedEmbeddingModel, path: str) -> FastembedVectorstore`
Loads a vector store from a JSON file.

## Performance Considerations

- **Memory Usage**: All embeddings are stored in memory, so consider the size of your document collection
- **Model Selection**: Smaller models are faster but may have lower quality embeddings
- **Batch Processing**: The `embed_documents` method processes documents in batches for efficiency

## Use Cases

- **Semantic Search**: Find documents similar to a query
- **Document Clustering**: Group similar documents together
- **Recommendation Systems**: Find similar items or content
- **Question Answering**: Retrieve relevant context for Q&A systems
- **Content Discovery**: Help users find related content

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](https://github.com/sauravniraula/fastembed_vectorstore/blob/main/LICENSE) file for details.

## Author

- **Saurav Niraula** - [sauravniraula](https://github.com/sauravniraula)
- Email: developmentsaurav@gmail.com

## Acknowledgments

- Built with [FastEmbed](https://github.com/Anush008/fastembed-rs) for efficient text embeddings
- Uses [PyO3](https://github.com/PyO3/pyo3) for Python-Rust bindings
- Inspired by the need for fast, lightweight vector storage solutions 