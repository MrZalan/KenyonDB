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
    MNIST_PATH=/opt/app/genn/mnist \
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
    optuna \
    pytest

# 5. Create app root and clone GeNN
RUN mkdir -p ${APP_PATH}
RUN git clone --branch master --recursive https://github.com/genn-team/genn.git ${GENN_PATH}

# 6. Set the Workdir to the cloned GeNN path
WORKDIR ${GENN_PATH}

# 7. Copy your local 'genn' folder content into the GeNN path
COPY genn/ . 

# 8. Add local pytest config for the mnist subproject
RUN mkdir -p ${MNIST_PATH} && \
    printf '%s\n' \
    '[pytest]' \
    'testpaths = tests' \
    'pythonpath = .' \
    > ${MNIST_PATH}/pytest.ini

# 9. Build PyGeNN
RUN python3 setup.py develop
ENV PYTHONPATH=${GENN_PATH}:${MNIST_PATH}

# 9. Permissions for Hugging Face (UID 1000)
RUN chmod -R 777 ${APP_PATH}

EXPOSE 7860

# Run app
CMD ["streamlit", "run", "mnist/app.py", "--server.address=0.0.0.0", "--server.port=7860"]

# For testing
#CMD ["/bin/bash"]