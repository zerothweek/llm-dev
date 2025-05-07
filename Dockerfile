# CUDA 11.8 and Ubuntu 22.04 based base image
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

LABEL maintainer="iis"
LABEL version="0.1"
LABEL description="first beta docker image to check nvidia gpu on docker container"

# Setting Environment Variables(Python version, timezone, koreanlanguage)
ARG PYTHON_VERSION=3.13.2 
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Seoul \
    LANG=ko_KR.UTF-8 \
    LANGUAGE=ko_KR:ko \
    LC_ALL=ko_KR.UTF-8

# Setting System Time Zone
RUN apt-get update && apt-get install -y --no-install-recommends \
    locales tzdata \
    && ln -fs /usr/share/zoneinfo/$TZ /etc/localtime \
    && dpkg-reconfigure --frontend noninteractive tzdata \
    && locale-gen $LC_ALL \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
    
### Python Installation & System Settings###
# Install dependencies for python installation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    curl \
    llvm \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    git \
    sudo \
    ca-certificates \
    vim \
    && apt-get clean && rm -rf /var/lib/apt/lists/*



# Download and build Python
RUN cd /usr/src \
    && wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz \
    && tar -xf Python-$PYTHON_VERSION.tgz \
    && cd Python-$PYTHON_VERSION \
    && ./configure --enable-optimizations --with-ensurepip=install \
    && make -j$(nproc) \
    && make altinstall \
    && cd .. \
    && rm -rf Python-$PYTHON_VERSION*
       
# Set default python and pip
RUN ln -s /usr/local/bin/python3.13 /usr/bin/python3 && \
    ln -s /usr/local/bin/pip3.13 /usr/bin/pip3 && \
    ln -s /usr/local/bin/python3.13 /usr/bin/python && \
    ln -s /usr/local/bin/pip3.13 /usr/bin/pip

RUN pip install --upgrade pip setuptools wheel
### Python Installation & System Settings###

# initial CMD
CMD ["/bin/bash"]

