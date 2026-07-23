FROM nvcr.io/nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ARG NB_USER=vscode
ARG NB_UID=1000

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3-pip python3-venv git curl ca-certificates build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install GitHub CLI so repo workflow commands are available inside the container.
RUN apt-get update && apt-get install -y --no-install-recommends gnupg && \
    mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | \
      gpg --dearmor -o /etc/apt/keyrings/githubcli-archive-keyring.gpg && \
    chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
      > /etc/apt/sources.list.d/github-cli.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends gh && \
    rm -rf /var/lib/apt/lists/*

# Create vscode user
RUN useradd -m -s /bin/bash -u ${NB_UID} ${NB_USER} || true

USER ${NB_USER}
WORKDIR /workspace/kortex

ENV PATH="/home/${NB_USER}/.local/bin:${PATH}"
