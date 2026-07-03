#!/bin/bash
# Install all extra Python deps into the RDU venv from wheelhouse/.
# Run on s339 via snrdu. Idempotent — safe to re-run.
#
# PYTHONNOUSERSITE=1 is critical here, not just cosmetic: the venv is
# created with --system-site-packages, so without it `pip install` sees
# user-site (~/.local) packages as already-satisfied and silently skips
# installing into the venv itself — the package then vanishes at actual
# runtime, where dynamo/vllm's exec environment sets PYTHONNOUSERSITE=1
# and user-site is no longer on sys.path. (Discovered via annotated_doc
# landing in ~/.local instead of the venv during 2026-07-02 validation.)
set -euo pipefail
export PYTHONNOUSERSITE=1
REPO_ROOT=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)
VENV="$REPO_ROOT/.venv_rdu"
WHLHOUSE="$REPO_ROOT/wheelhouse"

source "$VENV/bin/activate"

echo "=== Installing extra deps from wheelhouse ==="
PKGS=(
    openai_harmony  docstring_parser  durationpy  email_validator
    h11  mcp  mistral_common  fastar  llguidance  pybase64
    sniffio  astor  dnspython  pydantic_settings  pyjwt
    python_multipart  sse_starlette  starlette  typing_inspection
    uvicorn  pydantic_extra_types  tiktoken  ijson  partial_json_parser
    watchfiles  anthropic  fastapi  outlines_core
    prometheus_fastapi_instrumentator  python_json_logger
    xgrammar  kubernetes  model_hosting_container_standards
    exceptiongroup  httpx_sse  tqdm  lm_format_enforcer pydantic_core
    pycountry  annotated_doc  interegular  jmespath  python_dotenv
    requests_oauthlib  websocket_client  redis  oauthlib
    asgiref  cffi  cryptography  google_auth  googleapis_common_protos
    grpcio  grpcio_reflection  httptools  importlib_metadata  json_logic
    opentelemetry_api  opentelemetry_exporter_otlp  opentelemetry_sdk
    opentelemetry_semantic_conventions  protobuf  pyasn1  pyasn1_modules
    pyprctl  rich_toolkit  rignore  shellingham  typer  websockets  zipp
)

for name in "${PKGS[@]}"; do
    whl=$(find "$WHLHOUSE" -name "${name}*.whl" 2>/dev/null | sort -V | tail -1 || true)
    if [ -n "$whl" ]; then
        pip install -q --no-deps --force-reinstall --no-cache-dir "$whl" && echo "  installed: $(basename "$whl")"
    else
        echo "  MISSING wheel: $name"
    fi
done

# jiter: two variants exist in wheelhouse (cp311, cp312) — this venv is
# cp311. `sort -V | tail -1` above would pick cp312, which pip then
# rejects as "not a supported wheel on this platform" (name-sorts after
# cp311 lexically). Pin to the cp311 build explicitly.
JITER_WHL=$(find "$WHLHOUSE" -name "jiter-*cp311*.whl" 2>/dev/null | head -1 || true)
if [ -n "$JITER_WHL" ]; then
    pip install -q --no-deps --force-reinstall --no-cache-dir "$JITER_WHL" && echo "  installed: $(basename "$JITER_WHL")"
else
    echo "  MISSING wheel: jiter (cp311)"
fi

# Upgrade pydantic + typing-extensions to satisfy version requirements.
# jiter already installed (cp311-pinned) above — re-matching "jiter-*.whl"
# here would pick the cp312 wheel again via sort -V.
for name in pydantic typing_extensions; do
    whl=$(find "$WHLHOUSE" -name "${name}-*.whl" 2>/dev/null | sort -V | tail -1 || true)
    [ -n "$whl" ] && pip install -q --no-deps --force-reinstall --no-cache-dir "$whl" && echo "  upgraded: $(basename "$whl")"
done

echo ""
echo "=== pip check ==="
pip check 2>&1 | grep -v "grpcio-reflection\|opencv\|protobuf\|msgpack\|prometheus-client\|aiohttp" | head -20 || true

echo ""
echo "=== Import test ==="
# `import dynamo.vllm` alone doesn't exercise the full entrypoint import
# graph (async engine, FastAPI, multimodal_utils) — `python -m dynamo.vllm`
# does, and is what launch/rdu_decode.sh actually runs.
python -c "from dynamo.vllm.main import main; print('dynamo.vllm.main: OK')"
