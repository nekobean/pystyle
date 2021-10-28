FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
LABEL maintainer "pystyle"

ENV LC_ALL C.UTF-8
ENV PATH "/root/.pyenv/bin:${PATH}"
ARG PYTHON_VERSION="3.9.7"
WORKDIR /root

# Install requirements
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    # pyenv requirements: https://github.com/pyenv/pyenv/wiki#suggested-build-environment
    build-essential \
    curl \
    git \
    libbz2-dev \
    libffi-dev \
    liblzma-dev \
    libncurses5-dev \
    libncursesw5-dev \
    libreadline-dev \
    libsqlite3-dev \
    libssl-dev \
    libxml2-dev \
    libxmlsec1-dev \
    llvm \
    make \
    tk-dev \
    wget \
    xz-utils \
    zlib1g-dev \
    # git clone need CA certificates
    ca-certificates \
    # OpenCV need libGL.so.1
    libgl1

# Install SSH server
RUN apt-get install -y --no-install-recommends openssh-server && \
    echo "root:root" | chpasswd && \
    sed -i "s/#PermitRootLogin prohibit-password/PermitRootLogin yes/" /etc/ssh/sshd_config && \
    sed -i "s/#PubkeyAuthentication yes/PubkeyAuthentication yes/" /etc/ssh/sshd_config

# Register public key
ADD id_rsa.pub /root/.ssh/
RUN mv /root/.ssh/id_rsa.pub /root/.ssh/authorized_keys && \
    chmod 700 /root/.ssh && \
    chmod 600 /root/.ssh/authorized_keys

# Install nodejs for JupyterLab extension
RUN curl -sL https://deb.nodesource.com/setup_current.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    rm -rf /var/lib/apt/lists/*

# Install pyenv and python 3
RUN git clone https://github.com/pyenv/pyenv.git ~/.pyenv && \
    pyenv install $PYTHON_VERSION && \
    pyenv global $PYTHON_VERSION

# Install JupyterLab
COPY requirements.txt .
RUN eval "$(pyenv init --path)" && \
    cat requirements.txt | xargs -n 1 pip install -U --no-cache-dir && \
    ipython profile create && \
    # fix issue about black https://github.com/psf/black/issues/1143
    mkdir -p /root/.cache/black/21.9b0/ && \
    rm requirements.txt
COPY .jupyter/ /root/.jupyter/
COPY ipython_kernel_config.py /root/.ipython/profile_default/
COPY .bashrc /root/

EXPOSE 22 8888
CMD eval "$(pyenv init --path)" && service ssh start && jupyter lab
