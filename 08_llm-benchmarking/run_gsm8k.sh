#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

if [[ ! -f .env ]]; then
  echo "Error: $script_dir/.env was not found." >&2
  exit 1
fi

# Export variables from .env, overriding any stale value in the parent shell.
set -a
# shellcheck disable=SC1091
source .env
set +a

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "Error: OPENAI_API_KEY is missing from $script_dir/.env." >&2
  exit 1
fi

exec "$script_dir/venv/bin/lm-eval" run \
  --model openai-chat-completions \
  --model_args "model=gpt-5.6-luna,num_concurrent=5,max_retries=5" \
  --tasks gsm8k_cot \
  --num_fewshot 8 \
  --apply_chat_template \
  --limit 20 \
  --output_path ./gsm8k_results \
  --log_samples
