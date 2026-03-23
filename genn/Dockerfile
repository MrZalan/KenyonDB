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
    APP_PATH=/opt/app \
    GENN_PATH=/opt/app/genn \
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
    resdag \
    plotly \
    umap-learn \
    networkx \
    scikit-learn \
    mlflow \
    optuna

# 5. Create app root and clone GeNN there
RUN mkdir -p ${APP_PATH}
RUN git clone --branch master --recursive https://github.com/genn-team/genn.git ${GENN_PATH}

# 6. Copy project files into the cloned GeNN tree
COPY . ${GENN_PATH}/

# 7. Build PyGeNN from the merged tree
WORKDIR ${GENN_PATH}
RUN python3 setup.py develop

ENV PYTHONPATH=${GENN_PATH}

EXPOSE 7860

CMD ["sh", "-c", "cd mnist && streamlit run app.py --server.address=0.0.0.0 --server.port=7860"]