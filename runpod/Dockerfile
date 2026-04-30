# Run lfiax/sir.py and the cli-anything-lfiax harness on a RunPod CUDA
# pod. Builds on RunPod's PyTorch CUDA 12.1 base so torch ships with
# matching CUDA runtime libraries; we add a CUDA-enabled JAX on top.
#
# Build:
#   cd /Users/eric/Downloads/lfiax-code
#   docker build -f runpod/Dockerfile -t lfiax-runpod:latest .
#
# On RunPod: select "Custom container" with this image, or use the
# default PyTorch 2.1.0 CUDA 12.1 template and run runpod/setup.sh
# inside the pod (skipping the Dockerfile entirely).

FROM runpod/pytorch:2.1.0-py3.10-cuda12.1.1-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    JAX_PLATFORMS="cuda" \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

WORKDIR /workspace

# System tools we'll want for monitoring + git.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        less \
        htop \
        nvtop \
        tmux \
    && rm -rf /var/lib/apt/lists/*

# Copy lfiax-code (this Dockerfile is built from the lfiax-code root).
COPY . /workspace/lfiax-code/

# Install JAX stack + lfiax editable + cli-anything-lfiax editable.
RUN pip install --upgrade pip wheel \
    && pip install -r /workspace/lfiax-code/runpod/requirements.txt \
    && pip install -e /workspace/lfiax-code \
    && pip install -e /workspace/lfiax-code/agent-harness

# Generate the test SDE pkl that sir.py loads from cwd. Without this
# file sir.py exits immediately at FileNotFoundError. On a CUDA host
# the SDE generation runs on torch GPU and finishes in seconds.
RUN cd /workspace/lfiax-code && python runpod/gen_test_pkl.py

WORKDIR /workspace/lfiax-code

# Default command: launch the SIR baseline sweep. Override with
# `docker run ... bash` to drop into a shell instead.
CMD ["bash", "runpod/run_sir_baselines.sh"]
