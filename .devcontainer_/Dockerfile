FROM nvcr.io/nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ARG NB_USER=vscode
ARG NB_UID=1000

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3-pip python3-venv git curl ca-certificates build-essential && \
    rm -rf /var/lib/apt/lists/*

# Create vscode user
RUN useradd -m -s /bin/bash -u ${NB_UID} ${NB_USER} || true

USER ${NB_USER}
WORKDIR /workspace/kortex

ENV PATH="/home/${NB_USER}/.local/bin:${PATH}"
