ARG BASE=12.6.0-devel-ubuntu24.04
FROM nvidia/cuda:${BASE}

ARG GENN_VER=4.8.0
LABEL maintainer="J.C.Knight@sussex.ac.uk" version=${GENN_VER}

# 1. Install system dependencies
RUN apt-get update && apt-get install -yq --no-install-recommends \
    python3-dev \
    python3-pip \
    swig \
    gosu \
    nano \
    libffi-dev \
    pkg-config \
    build-essential \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 2. Set python3 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# 3. Environment
ENV CUDA_PATH=/usr/local/cuda \
    GENN_PATH=/opt/genn \
    APP_PATH=/opt/app \
    PYTHONUNBUFFERED=1

# 4. Install Python dependencies
RUN python -m pip install --no-cache-dir --break-system-packages \
    numpy \
    jupyter \
    matplotlib \
    psutil \
    pybind11 \
    mnist \
    tqdm \
    pkgconfig \
    torch \
    torchvision \
    opencv-python-headless \
    streamlit \
    streamlit-drawable-canvas \
    resdag

# 5. Clone and build GeNN
RUN git clone --branch v${GENN_VER} --recursive https://github.com/genn-team/genn.git ${GENN_PATH}
WORKDIR ${GENN_PATH}
RUN python3 setup.py develop

# 6. Copy project files
COPY . ${APP_PATH}/
WORKDIR ${APP_PATH}

ENV PYTHONPATH=${APP_PATH}

EXPOSE 8501

CMD ["/bin/bash"]