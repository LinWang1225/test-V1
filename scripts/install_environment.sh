#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PATH="$ROOT/.venv/bin:/home/wanglin/.cargo/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-12.6
export TORCH_CUDA_ARCH_LIST=8.6
export MAX_JOBS=12
export NVCC_THREADS=2
export UV_HTTP_TIMEOUT=600
unset NO_PROXY
source /opt/lab-server-ops/lso_user/script_out/out.sh
export no_proxy=localhost,127.0.0.1,10.177.73.90
lso_proxy_on 10.177.73.90 7890
export VLLM_CUTLASS_SRC_DIR="$ROOT/src/KVarN/.deps/cutlass-src"
export VLLM_FLASH_ATTN_SRC_DIR="$ROOT/src/KVarN/.deps/vllm-flash-attn-src"
export DEEPGEMM_SRC_DIR="$ROOT/src/KVarN/.deps/deepgemm-src"
export FLASH_MLA_SRC_DIR="$ROOT/src/KVarN/.deps/flashmla-src"
export QUTLASS_SRC_DIR="$ROOT/src/KVarN/.deps/qutlass-src"
export TRITON_KERNELS_SRC_DIR="$ROOT/src/KVarN/.deps/triton_kernels-src/python/triton_kernels/triton_kernels"
uv pip install --python "$ROOT/.venv/bin/python" -r locks/serve-cu126.txt --index https://pypi.org/simple --default-index https://download.pytorch.org/whl/cu126 --index-strategy unsafe-first-match
uv pip install --python "$ROOT/.venv/bin/python" -r src/KVarN/requirements/build/cuda.txt -r src/KVarN/requirements/cuda.txt -r src/KVarN/requirements/lint.txt -c locks/serve-cu126.txt
uv pip install --python "$ROOT/.venv/bin/python" --no-build-isolation --no-deps -e src/KVarN
(cd src/KVarN && pre-commit install)
uv pip check --python "$ROOT/.venv/bin/python"
uv pip freeze --python "$ROOT/.venv/bin/python" > locks/installed.freeze.txt
