import json
import random
from pathlib import Path

from dotenv import load_dotenv
from deepeval.synthesizer import Synthesizer
from langchain_text_splitters.character import RecursiveCharacterTextSplitter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_PATH = PROJECT_ROOT / "goldens" / "retriever_deepeval_goldens.json"

load_dotenv(PROJECT_ROOT / ".env", override=True)

# --- reuse your own VTT cleaning + chunking (same as the retriever) ---
def load_chunks():
    texts = []
    for path in DATA_DIR.glob("*.vtt"):
        with path.open(encoding="utf-8") as f:
            lines = [ln.strip() for ln in f
                     if ln.strip() and ln.strip() != "WEBVTT" and "-->" not in ln]
        texts.append(" ".join(lines))
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    return splitter.split_text("\n\n".join(texts))


# --- generate ---
chunks = load_chunks()
sample = random.sample(chunks, min(15, len(chunks)))     # ~12 chunks -> keep the set small
contexts = [[c] for c in sample]                          # each context = one chunk

synthesizer = Synthesizer(model="gpt-4.1-mini")                # the generator/critic model -- pin it
goldens = synthesizer.generate_goldens_from_contexts(
    contexts=contexts,
    include_expected_output=True,       # <-- THIS gives you the ideal_answer
    max_goldens_per_context=1,          # 1 question per chunk -> ~12 goldens
)


# --- convert to YOUR schema (id / query / ideal_answer / source) ---
rows = []
for i, g in enumerate(goldens, 1):
    rows.append({
        "id": f"g{i:03d}",
        "query": g.input,
        "ideal_answer": g.expected_output,
        "source": "TODO-verify",        # Synthesizer won't know the session -- you fill this
    })

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with OUTPUT_PATH.open("w", encoding="utf-8") as f:
    json.dump(rows, f, indent=2, ensure_ascii=False)

print(f"wrote {len(rows)} DRAFT goldens -> {OUTPUT_PATH}")
print("!! REVIEW EVERY ONE before using: check grounding, trim padding, fix leading questions.")
