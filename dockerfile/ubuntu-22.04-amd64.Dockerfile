# TT-METAL UBUNTU 22.04 AMD64 DOCKERFILE
FROM public.ecr.aws/ubuntu/ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG UBUNTU_VERSION=22.04
ENV DOXYGEN_VERSION=1.9.6
ENV CCACHE_TEMPDIR=/tmp/ccache

# Use a newer version of CMake than what is available from Canonical for 22.04
RUN apt -y update \
    && apt install -y --no-install-recommends ca-certificates gpg wget \
    && wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null \
    && echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ jammy main' | tee /etc/apt/sources.list.d/kitware.list >/dev/null \
    && rm -rf /var/lib/apt/lists/*

RUN apt update -y && apt install software-properties-common gpg-agent -y

# add custom repo
RUN add-apt-repository ppa:deadsnakes/ppa

# Install build and runtime deps
COPY /scripts/docker/requirements-${UBUNTU_VERSION}.txt /opt/tt_metal_infra/scripts/docker/requirements.txt
RUN apt-get -y update \
    && xargs -a /opt/tt_metal_infra/scripts/docker/requirements.txt apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Install dev deps
COPY /scripts/docker/requirements_dev.txt /opt/tt_metal_infra/scripts/docker/requirements_dev.txt
RUN apt-get -y update \
    && xargs -a /opt/tt_metal_infra/scripts/docker/requirements_dev.txt apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

## Test Related Dependencies
COPY /scripts/docker/install_test_deps.sh /opt/tt_metal_infra/scripts/docker/install_test_deps.sh
RUN /bin/bash /opt/tt_metal_infra/scripts/docker/install_test_deps.sh ${DOXYGEN_VERSION}

# Copy remaining convenience scripts
COPY /scripts /opt/tt_metal_infra/scripts
COPY build_metal.sh /scripts/build_metal.sh

# Setup Env variables to setup Python Virtualenv - Install TT-Metal Python deps
ENV TT_METAL_INFRA_DIR=/opt/tt_metal_infra
ENV PYTHON_ENV_DIR=${TT_METAL_INFRA_DIR}/tt-metal/python_env

# Disable using venv since this is isolated in a docker container
# RUN python3 -m venv $PYTHON_ENV_DIR
# ENV PATH="$PYTHON_ENV_DIR/bin:$PATH"

# Create directories for infra
RUN mkdir -p ${TT_METAL_INFRA_DIR}/tt-metal/docs/
RUN mkdir -p ${TT_METAL_INFRA_DIR}/tt-metal/tests/sweep_framework/
RUN mkdir -p ${TT_METAL_INFRA_DIR}/tt-metal/tt_metal/python_env/

# Copy requirements from tt-metal folders with requirements.txt docs
COPY /docs/requirements-docs.txt ${TT_METAL_INFRA_DIR}/tt-metal/docs/.
# Copy requirements from tt-metal folders for sweeps (requirements-sweeps.txt)
COPY /tests/sweep_framework/requirements-sweeps.txt ${TT_METAL_INFRA_DIR}/tt-metal/tests/sweep_framework/.
COPY /tt_metal/python_env/* ${TT_METAL_INFRA_DIR}/tt-metal/tt_metal/python_env/.
RUN python3 -m pip config set global.extra-index-url https://download.pytorch.org/whl/cpu \
    && python3 -m pip install setuptools wheel

RUN python3 -m pip install -r ${TT_METAL_INFRA_DIR}/tt-metal/tt_metal/python_env/requirements-dev.txt
RUN python3 -m pip install -r ${TT_METAL_INFRA_DIR}/tt-metal/docs/requirements-docs.txt

# Install Clang-17
RUN cd $TT_METAL_INFRA_DIR \
    && wget https://apt.llvm.org/llvm.sh \
    && chmod u+x llvm.sh \
    && ./llvm.sh 17

# Install compatible gdb debugger for clang-17
RUN cd $TT_METAL_INFRA_DIR \
    && wget https://ftp.gnu.org/gnu/gdb/gdb-14.2.tar.gz \
    && tar -xvf gdb-14.2.tar.gz \
    && cd gdb-14.2 \
    && ./configure \
    && make -j$(nproc)
ENV PATH="$TT_METAL_INFRA_DIR/gdb-14.2/gdb:$PATH"

# Can only be installed after Clang-17 installed
RUN apt-get -y update \
    && apt-get install -y --no-install-recommends \
    libc++-17-dev \
    libc++abi-17-dev \
    clang-tidy-17 \
    && rm -rf /var/lib/apt/lists/*

# Setup Env variables to setup Python Virtualenv - Install TT-Metal Python deps
ENV TT_METAL_INFRA_DIR=/opt/tt_metal_infra
ENV PYTHON_ENV_DIR=${TT_METAL_INFRA_DIR}/tt-metal/python_env

# Disable using venv since this is isolated in a docker container
# RUN python3 -m venv $PYTHON_ENV_DIR
# ENV PATH="$PYTHON_ENV_DIR/bin:$PATH"

# Copy requirements from tt-metal folders with requirements.txt docs
COPY /docs/requirements-docs.txt ${TT_METAL_INFRA_DIR}/tt-metal/docs/.
# Copy requirements from tt-metal folders for sweeps (requirements-sweeps.txt)
COPY /tests/sweep_framework/requirements-sweeps.txt ${TT_METAL_INFRA_DIR}/tt-metal/tests/sweep_framework/.
COPY /tt_metal/python_env/* ${TT_METAL_INFRA_DIR}/tt-metal/tt_metal/python_env/.
RUN python3 -m pip config set global.extra-index-url https://download.pytorch.org/whl/cpu \
    && python3 -m pip install setuptools wheel

RUN python3 -m pip install -r ${TT_METAL_INFRA_DIR}/tt-metal/tt_metal/python_env/requirements-dev.txt
RUN python3 -m pip install -r ${TT_METAL_INFRA_DIR}/tt-metal/docs/requirements-docs.txt

# Set python 3.11 and gcc-12 to be default
# RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 10
# RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 11

RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 11
RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 12

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 11
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 12

# Ccache is not in requirements_dev.txt because 20.04's version is too old for remote_storage support.
# When we drop 20.04, can put it in requirements_dev.txt instead of here.
RUN wget -O /tmp/ccache.tar.xz https://github.com/ccache/ccache/releases/download/v4.10.2/ccache-4.10.2-linux-x86_64.tar.xz && \
    tar -xf /tmp/ccache.tar.xz -C /usr/local/bin --strip-components=1 && \
    rm /tmp/ccache.tar.xz
RUN ccache --version

RUN mkdir -p /usr/app

# CMD ["tail", "-f", "/dev/null"]
