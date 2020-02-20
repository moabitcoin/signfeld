FROM nvidia/cuda:10.1-cudnn7-devel

WORKDIR /usr/src/app

ENV LANG="C.UTF-8" LC_ALL="C.UTF-8" PATH="/opt/venv/bin:$PATH" PIP_NO_CACHE_DIR="false" CFLAGS="-mavx2 -mf16c" CXXFLAGS="-mavx2 -mf16c"

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev git libglib2.0-dev libsm6 libxext6 libxrender-dev \
    libpng-dev libjpeg-dev build-essential pkg-config \
    wget curl automake libtool && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv && \
    python3 -m pip install pip==19.3.1 pip-tools==4.2.0

COPY requirements.txt .

RUN python3 -m piptools sync

RUN python3 -m pip install git+https://github.com/cocodataset/cocoapi.git@636becdc73d54283b3aac6d4ec363cffbb6f9b20#subdirectory=PythonAPI
RUN FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="Maxwell;Pascal;Volta;Turing" python3 -m pip install git+https://github.com/facebookresearch/detectron2@6a422717df9480f23b062be0777f56a227cad33a#egg=detectron2

COPY . .
RUN python3 -m pip install .[trainer]

ENTRYPOINT ["/bin/bash"]
