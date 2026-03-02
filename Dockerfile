FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common \
        ca-certificates \
        wget \
        curl \
        git \
        build-essential \
        screen  \
        vim  \
        tmux  \
        htop \
        net-tools \
        iputils-ping \
        iproute2 \
        lsof && \
    echo "Installed basic packages."


RUN add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update


RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends python3.12

RUN apt-cache policy python3.12-distutils


RUN apt-get install -y --no-install-recommends python3.12-dev


RUN curl -sSL https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.12 get-pip.py --no-cache-dir && \
    rm get-pip.py


RUN python3.12 -m pip install --upgrade pip setuptools




RUN add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt-cache policy python3.12 && \
    echo "Added deadsnakes PPA."


RUN apt-get update && \
    echo "Updated package lists after adding PPA."



RUN apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-dev \
        python3.12-distutils && \
    echo "Installed Python 3.12 and related packages."


RUN apt-get clean && rm -rf /var/lib/apt/lists/*


RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1


RUN wget -qO- https://bootstrap.pypa.io/get-pip.py | python


RUN python -m pip install --upgrade pip && \
    python -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

RUN python -m pip install torch==2.5.1 torchvision torchaudio \
    --extra-index-url https://download.pytorch.org/whl/cu124

RUN python -m pip install modelscope==1.25.0 \
    transformers==4.51.3 \
    accelerate==1.6.0 \
    datasets==3.5.1 \
    peft==0.15.2 \
    swanlab==0.5.7 \
    tqdm==4.66.4 \
    pandas==2.2.2 \
    python-Levenshtein==0.27.1 \
    evaluate==0.4.1 \
    sentence-transformers==2.7.0 \
    jupyterlab \
    deepspeed==0.14.4


RUN apt-get update && \
    apt-get install -y openssh-server && \
    mkdir /var/run/sshd

EXPOSE 22

WORKDIR /workspace

CMD ["/bin/bash"]
