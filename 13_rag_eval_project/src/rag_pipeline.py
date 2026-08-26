"""End-to-end retrieval and generation pipeline.

This module can be launched either as ``python -m src.rag_pipeline`` (preferred)
or directly as ``python src/rag_pipeline.py``.
"""

import sys
from pathlib import Path


# When this file is executed directly, Python adds ``src/`` rather than the
# project directory to sys.path. Add the project directory so the ``src``
# package can still be imported. Module execution already has the correct path.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.generator import generate
from src.reranker import RerankingRetriever


class RagPipeline:
    def __init__(self, fetch_k=10, top_k=5):
        # one retriever instance — loads the store + reranker model once
        self.retriever = RerankingRetriever(fetch_k=fetch_k, top_k=top_k)

    def invoke(self, query: str) -> dict:
        # 1. RETRIEVE: over-fetch then rerank down to top_k Documents
        docs = self.retriever.invoke(query)

        # 2. UNPACK: generator wants list[str], the triad wants the same strings
        context = [doc.page_content for doc in docs]

        # 3. GENERATE: grounded answer from the retrieved context
        answer = generate(query, context)

        # return all three legs of the triad so the eval harness can score them
        return {
            "query": query,
            "context": context,
            "answer": answer,
        }


# quick manual smoke test: python -m src.rag_pipeline
if __name__ == "__main__":
    rag = RagPipeline()
    result = rag.invoke("what is drift and why does it matter after deployment?")
    print("QUERY:  ", result["query"])
    print("ANSWER: ", result["answer"])
    print("\nCONTEXT CHUNKS:")
    for i, chunk in enumerate(result["context"]):
        print(f"  [{i}] {chunk[:120]}...")
