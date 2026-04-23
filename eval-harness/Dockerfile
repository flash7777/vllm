# syntax=docker/dockerfile:1
# MultiQuant evaluation harness (model-agnostic).
# Runs lm-eval-harness against any OpenAI-compat API (vLLM, llama.cpp
# server, TGI, OpenAI-proper, …) via the `local-completions` backend.
#
# Intentionally CPU-only / tiny (~300 MB). GPU work happens on the
# remote server under test; this container is a pure client for
# dataset replay, prompt templating, metric aggregation.
#
# Build:  eval-harness/build.sh
# Run:    see bench.lm-eval (top-level wrapper)

FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TRANSFORMERS_VERBOSITY=warning

RUN pip install --no-cache-dir \
    "lm-eval[api]==0.4.11" \
    "numpy>=1.26,<3" \
    "datasets>=2.19" \
    "scipy>=1.11" \
    "tqdm" \
    "requests" \
    "transformers>=4.45" \
    "tokenizers>=0.20"

WORKDIR /opt/eval
ENV PYTHONPATH=/opt/eval

CMD ["lm_eval", "--help"]
