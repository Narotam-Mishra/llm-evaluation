import json
import sys
from pathlib import Path

# Allow this file to be run directly from any working directory. When Python
# executes a file inside evals/, it otherwise adds only evals/ to sys.path and
# cannot see the sibling src/ package.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric, ContextualPrecisionMetric

from src.retriever import build_retriever

load_dotenv(PROJECT_ROOT / ".env", override=True)

GOLDEN_PATH = PROJECT_ROOT / "goldens" / "retriever_goldens.json"
JUDGE_MODEL = "gpt-4.1-mini"  
THRESHOLD = 0.7


# 1. LOAD the golden set --- the fixed, human-authored truth
with GOLDEN_PATH.open(encoding="utf-8") as f:
    goldens = json.load(f)


# 2. RUN THE RETRIEVER on each question to fill retrieval_context,
#    then build one test case per golden.
retriever = build_retriever()

test_cases = []

for g in goldens:
    retrieved = retriever.invoke(g["query"])
    retrieval_context = [doc.page_content for doc in retrieved]

    test_cases.append(
        LLMTestCase(
            input=g["query"],
            expected_output=g["ideal_answer"],
            retrieval_context=retrieval_context,
            actual_output="(generator not evaluated in this run)",
        )
    )


# 3. THE METRICS --- recall (did we miss?) and precision (ranked well?)
metrics = [
    ContextualRecallMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
    ContextualPrecisionMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
]


# 4. EVALUATE --- every metric on every case, batched + parallel, with a printed report
evaluate(
    test_cases=test_cases,
    metrics=metrics,
    hyperparameters={
        "retriever": "base_k5",          # vs "reranked" when you swap it in
        "embedding_model": "text-embedding-3-large",
        "chunk_size": 1000,
        "chunk_overlap": 150,
        "top_k": 5,
        "judge_model": JUDGE_MODEL,
        "golden_set": str(GOLDEN_PATH),
    },
)
