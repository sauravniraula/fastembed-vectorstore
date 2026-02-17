import unittest

import fastembed_vectorstore as fvs


class DummyTextEmbedding:
    def __init__(self, model_name=None, cache_dir=None, show_progress=None):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.show_progress = show_progress

    @staticmethod
    def list_supported_models():
        return [{"model_name": "BGESmallENV15"}]

    def embed(self, documents, *_args, **_kwargs):
        for text in documents:
            lowered = text.lower()
            python_hits = lowered.count("python")
            rust_hits = lowered.count("rust")
            length = len(lowered)
            yield [float(python_hits), float(rust_hits), float(length)]


class VectorstoreTestCase(unittest.TestCase):
    def setUp(self):
        self._original_embedding = fvs.TextEmbedding
        fvs.TextEmbedding = DummyTextEmbedding

    def tearDown(self):
        fvs.TextEmbedding = self._original_embedding

    def test_search_returns_expected_result(self):
        vectorstore = fvs.FastembedVectorstore(fvs.FastembedEmbeddingModel.BGESmallENV15)
        documents = [
            "The quick brown fox jumps over the lazy dog",
            "A quick brown dog jumps over the lazy fox",
            "The lazy fox sleeps while the quick brown dog watches",
            "Python is a programming language",
            "Rust is a systems programming language",
        ]
        vectorstore.embed_documents(documents)
        results = vectorstore.search("What is Python?", n=1)
        self.assertEqual(results[0][0], "Python is a programming language")


if __name__ == "__main__":
    unittest.main()
