# RL runtime: DeepMind Lab + PyTorch.
# Build:  docker build -f Dockerfile.rl -t dmlab-rl:latest .            (CUDA)
#         docker build -f Dockerfile.rl -t dmlab-rl:cpu \
#           --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu .
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128
FROM dmlab:latest
ARG TORCH_INDEX
RUN apt-get update && apt-get install -y --no-install-recommends python3-pip \
    && rm -rf /var/lib/apt/lists/*
# numpy must stay <2: the deepmind_lab bindings in the base image were
# compiled against the numpy 1.x ABI.
RUN pip3 install --no-cache-dir torch --index-url ${TORCH_INDEX} \
    && pip3 install --no-cache-dir 'numpy<2' scipy scikit-learn \
    && python3 -c 'import numpy, deepmind_lab, torch'
WORKDIR /workspace
