import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv(override=True)  # loads OPENAI_API_KEY from .env

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
DB_DIR = PROJECT_DIR / "chroma_store"


# 1. LOAD ---- read each transcript, throw away the VTT timestamps
def load_transcripts():
    docs = []
    for path in sorted(DATA_DIR.glob("*.vtt")):
        lines = []
        with path.open(encoding="utf-8") as transcript:
            for line in transcript:
                line = line.strip()
                if not line or line == "WEBVTT" or "-->" in line:
                    continue
                lines.append(line)
        text = " ".join(lines)

        match = re.search(r"Session[ _]*(\d+)", path.name)
        if not match:
            raise ValueError(f"Could not determine session number from {path.name}")

        if text:
            docs.append(
                Document(page_content=text, metadata={"session": match.group(1)})
            )

    if not docs:
        raise FileNotFoundError(
            f"No non-empty VTT transcripts found in {DATA_DIR}. "
            "Add transcript files before building the vector store."
        )

    return docs


# 2. BUILD ---- chunk, embed once, and keep it on disk so we don't re-embed
def load_store():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    if (DB_DIR / "chroma.sqlite3").exists():
        store = Chroma(
            persist_directory=str(DB_DIR), embedding_function=embeddings
        )
        if store._collection.count() > 0:
            return store

    docs = load_transcripts()

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=750,
        chunk_overlap=100,
    ).split_documents(docs)

    if not chunks:
        raise ValueError("Transcript loading succeeded, but produced no text chunks.")

    return Chroma.from_documents(
        chunks, embeddings, persist_directory=str(DB_DIR)
    )


def build_retriever():
    return load_store().as_retriever(search_kwargs={"k": 5})


# 3. TRY IT ---- python src/retriever.py
if __name__ == "__main__":

    retriever = build_retriever()

    results = retriever.invoke("what is regression testing?")
    
    for r in results:
        print(f"[Session {r.metadata['session']}] {r.page_content[:150]}...\n")
