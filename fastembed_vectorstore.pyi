from typing import List, Tuple, Optional, Union
import os
from enum import Enum

class FastembedEmbeddingModel(Enum):
    """Enumeration of available embedding models."""
    AllMiniLML6V2: "FastembedEmbeddingModel"
    AllMiniLML6V2Q: "FastembedEmbeddingModel"
    AllMiniLML12V2: "FastembedEmbeddingModel"
    AllMiniLML12V2Q: "FastembedEmbeddingModel"
    BGEBaseENV15: "FastembedEmbeddingModel"
    BGEBaseENV15Q: "FastembedEmbeddingModel"
    BGELargeENV15: "FastembedEmbeddingModel"
    BGELargeENV15Q: "FastembedEmbeddingModel"
    BGESmallENV15: "FastembedEmbeddingModel"
    BGESmallENV15Q: "FastembedEmbeddingModel"
    NomicEmbedTextV1: "FastembedEmbeddingModel"
    NomicEmbedTextV15: "FastembedEmbeddingModel"
    NomicEmbedTextV15Q: "FastembedEmbeddingModel"
    ParaphraseMLMiniLML12V2: "FastembedEmbeddingModel"
    ParaphraseMLMiniLML12V2Q: "FastembedEmbeddingModel"
    ParaphraseMLMpnetBaseV2: "FastembedEmbeddingModel"
    BGESmallZHV15: "FastembedEmbeddingModel"
    BGELargeZHV15: "FastembedEmbeddingModel"
    ModernBertEmbedLarge: "FastembedEmbeddingModel"
    MultilingualE5Small: "FastembedEmbeddingModel"
    MultilingualE5Base: "FastembedEmbeddingModel"
    MultilingualE5Large: "FastembedEmbeddingModel"
    MxbaiEmbedLargeV1: "FastembedEmbeddingModel"
    MxbaiEmbedLargeV1Q: "FastembedEmbeddingModel"
    GTEBaseENV15: "FastembedEmbeddingModel"
    GTEBaseENV15Q: "FastembedEmbeddingModel"
    GTELargeENV15: "FastembedEmbeddingModel"
    GTELargeENV15Q: "FastembedEmbeddingModel"
    ClipVitB32: "FastembedEmbeddingModel"
    JinaEmbeddingsV2BaseCode: "FastembedEmbeddingModel"

class FastembedVectorstore:
    """A Rust-based in-memory vector store with fastembed integration."""
    
    def __init__(
        self,
        model: FastembedEmbeddingModel,
        show_download_progress: Optional[bool] = ...,
        cache_directory: Union[str, os.PathLike[str], None] = ...,
    ) -> None:
        """
        Initialize a new vector store with the specified embedding model.
        
        Args:
            model: The embedding model to use for generating embeddings
            show_download_progress: Whether to show model download progress (default: True)
            cache_directory: Directory to cache model files (default: fastembed_cache)
        """
        ...
    
    @classmethod
    def load(cls, model: FastembedEmbeddingModel, path: str) -> "FastembedVectorstore":
        """
        Load a vector store from a JSON file.
        
        Args:
            model: The embedding model to use
            path: Path to the JSON file containing saved embeddings
            
        Returns:
            A FastembedVectorstore instance loaded with embeddings from the file
            
        Raises:
            Exception: If the file doesn't exist or cannot be parsed
        """
        ...
    
    def embed_documents(self, documents: List[str]) -> bool:
        """
        Embed a list of documents and store them in the vector store.
        
        Args:
            documents: List of text documents to embed
            
        Returns:
            True if embedding was successful, False otherwise
        """
        ...
    
    def search(self, query: str, n: int) -> List[Tuple[str, float]]:
        """
        Search for similar documents using cosine similarity.
        
        Args:
            query: The search query text
            n: Number of top results to return
            
        Returns:
            List of tuples containing (document_text, similarity_score) sorted by similarity
        """
        ...
    
    def save(self, path: str) -> bool:
        """
        Save the vector store embeddings to a JSON file.
        
        Args:
            path: Path where to save the JSON file
            
        Returns:
            True if saving was successful, False otherwise
        """
        ... 