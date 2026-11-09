#!/usr/bin/env bash
set -euo pipefail

ROOT="${MEWHELP_RUNTIME_ROOT:-/data/home/mewhelp-runtime}"
RELEASES="$ROOT/releases"
DATA="$ROOT/data"
SERVICE="${MEWHELP_SERVICE_NAME:-mewhelp-demo.service}"
PORT="${MEWHELP_PORT:-8000}"
DEMO_DATABASE="${MEWHELP_DEMO_DATABASE:-/data/workspace/mewhelp-runtime/mewhelp.db}"
target="${1:-}"

case "$target" in
  pure)
    database="$DATA/pure.db"
    llm_mode="openai"
    llm_model="${MEWHELP_LOCAL_LLM_MODEL:-qwen2.5:1.5b}"
    llm_api_key="${MEWHELP_LOCAL_LLM_API_KEY:-ollama-local}"
    llm_base_url="${MEWHELP_LOCAL_LLM_BASE_URL:-http://127.0.0.1:11434/v1}"
    ;;
  demo)
    database="$DEMO_DATABASE"
    llm_mode="mock"
    llm_model="gpt-5-mini"
    llm_api_key=""
    llm_base_url=""
    ;;
  *) echo "usage: $0 pure|demo" >&2; exit 2 ;;
esac

release="$RELEASES/$target"
if [[ ! -f "$release/src/mewhelp/api/app.py" ]]; then
  echo "release not found: $release" >&2
  exit 1
fi

if [[ "$target" == "pure" ]] && ! curl -fsS --max-time 3 http://127.0.0.1:11434/api/tags >/dev/null; then
  echo "local Ollama is unavailable; start mewhelp-ollama before switching to pure" >&2
  exit 1
fi

mkdir -p "$DATA"
previous_release="$(readlink -f "$ROOT/current" 2>/dev/null || true)"
previous_env="$ROOT/active.env.previous.$$"
if [[ -f "$ROOT/active.env" ]]; then
  cp -p "$ROOT/active.env" "$previous_env"
fi

link_tmp="$ROOT/current.tmp.$$"
ln -s "$release" "$link_tmp"
mv -Tf "$link_tmp" "$ROOT/current"

env_tmp="$ROOT/active.env.tmp.$$"
cat > "$env_tmp" <<ENV
PYTHONPATH=$release/src
MEWHELP_DATABASE_URL=sqlite:///$database
MEWHELP_LLM_MODE=$llm_mode
MEWHELP_LLM_MODEL=$llm_model
MEWHELP_OPENAI_API_KEY=$llm_api_key
MEWHELP_OPENAI_BASE_URL=$llm_base_url
MEWHELP_EMBEDDING_MODE=mock
MEWHELP_RERANKER_MODE=mock
MEWHELP_VECTOR_STORE_MODE=memory
MEWHELP_LANGFUSE_ENABLED=false
OPENAI_API_KEY=
OPENAI_BASE_URL=
ENV
chmod 600 "$env_tmp"
mv -Tf "$env_tmp" "$ROOT/active.env"

systemctl --user daemon-reload
systemctl --user restart "$SERVICE"

healthy=0
for _ in $(seq 1 30); do
  if curl -fsS --max-time 2 "http://127.0.0.1:$PORT/api/health" >/dev/null; then
    healthy=1
    break
  fi
  sleep 1
done

if [[ "$healthy" -ne 1 ]]; then
  echo "new release failed health check; restoring previous release" >&2
  if [[ -n "$previous_release" ]]; then
    restore_link="$ROOT/current.restore.$$"
    ln -s "$previous_release" "$restore_link"
    mv -Tf "$restore_link" "$ROOT/current"
  fi
  if [[ -f "$previous_env" ]]; then
    mv -Tf "$previous_env" "$ROOT/active.env"
  fi
  systemctl --user daemon-reload
  systemctl --user restart "$SERVICE" || true
  exit 1
fi

rm -f "$previous_env"
printf 'active=%s\nrelease=%s\ndatabase=%s\n' "$target" "$release" "$database"
printf 'health='
curl -fsS "http://127.0.0.1:$PORT/api/health"
printf '\n'
