ARG PYTORCH="1.5"
ARG CUDA="10.1"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    ninja-build \
    libglib2.0-0 \
    libsm6 \
    libxrender-dev \
    libxext6 \
    locales \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

RUN pip install --no-cache-dir --upgrade pip

COPY ./requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Install mmdetection
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
ENV FORCE_CUDA="1"
RUN git clone https://github.com/open-mmlab/mmdetection /tmp/mmdetection \
    && cd /tmp/mmdetection \
    && git checkout 38dfa875c048207fd46b8cd2b7ccafd5239b4a4e \
    && pip install --no-cache-dir "git+https://github.com/open-mmlab/cocoapi.git#subdirectory=pycocotools" \
    && pip install --no-cache-dir /tmp/mmdetection \
    && rm -r /tmp/mmdetection

RUN pip install --no-cache-dir mmcv==0.6.2

ENV PROJECT_ROOT /global-wheat-detection
ENV PYTHONPATH "${PYTHONPATH}:${PROJECT_ROOT}"
WORKDIR /global-wheat-detection
