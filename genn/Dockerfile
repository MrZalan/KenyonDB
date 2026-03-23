ARG BASE=12.6.0-devel-ubuntu24.04
FROM nvidia/cuda:${BASE}

ARG GENN_VER=4.8.0
LABEL maintainer="J.C.Knight@sussex.ac.uk" version=${GENN_VER}

# 1. Install system dependencies
RUN apt-get update && apt-get install -yq --no-install-recommends \
    python3-dev python3-pip swig gosu nano libffi-dev pkg-config build-essential git

# 2. Set python3 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# 3. Install Python dependencies
RUN python -m pip install numpy jupyter matplotlib psutil pybind11 mnist tqdm pkgconfig --break-system-packages && \
    python -m pip install torch torchvision opencv-python-headless streamlit streamlit-drawable-canvas resdag --break-system-packages

# 4. Setup GeNN
ENV CUDA_PATH=/usr/local/cuda GENN_PATH=/opt/genn
COPY . ${GENN_PATH}
WORKDIR ${GENN_PATH}

# 5. Build the PyGeNN backend
RUN python3 setup.py develop

EXPOSE 8501
ENTRYPOINT ["/opt/genn/bin/genn-docker-entrypoint.sh"]
CMD ["/bin/bash"]