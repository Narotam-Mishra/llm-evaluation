"""
First test (ChatOpenRouter version): send ONE question to ONE model via
OpenRouter, print the raw SQL. No loop, no scoring, no cleaning.

Setup:
  pip install langchain-openrouter
  Add OPENROUTER_API_KEY=your_key to a .env file next to this script,
  then choose the MODEL below.
"""

import os
from pathlib import Path

from langchain_openrouter import ChatOpenRouter
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent
load_dotenv(SCRIPT_DIR / ".env", override=True)

# ---------- FILL THESE IN ----------
MODEL   = "openai/gpt-4o"          # OpenRouter model slug
# -----------------------------------

# Prefer the documented variable name. API_KEY is retained so the existing
# local .env file continues to work; rename it to OPENROUTER_API_KEY when able.
api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("API_KEY")
if not api_key:
    raise RuntimeError(
        "Missing OpenRouter API key. Add this line to "
        f"{SCRIPT_DIR / '.env'}:\nOPENROUTER_API_KEY=your_openrouter_key"
    )

# 1. Read the schema from disk
with open(SCRIPT_DIR / "schema.sql", encoding="utf-8") as f:
    schema = f.read().strip()

# 2. The one question we're testing
question = "Who scored the most runs in 2024?"

# 3. Build the prompt (schema + question)
system_msg = (
    "You are a text-to-SQL generator. "
    "Given a database schema and a question, return a single SQL query that answers it. "
    "Use SQLite syntax. Return only the SQL query."
)
user_msg = f"Schema:\n{schema}\n\nQuestion: {question}\n\nSQL:"

# 4. Connect via ChatOpenRouter
llm = ChatOpenRouter(
    api_key=api_key,
    model=MODEL,
    temperature=0,
    max_tokens=300
)

# 5. Send it and print what comes back
messages = [
    SystemMessage(content=system_msg),
    HumanMessage(content=user_msg),
]
response = llm.invoke(messages)

raw_sql = response.content

print("=" * 60)
print("QUESTION:", question)
print("=" * 60)
print("RAW MODEL OUTPUT:")
print(raw_sql)
print("=" * 60)
