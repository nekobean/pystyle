FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04
LABEL maintainer "pystyle"

ARG DOCKER_UID=1000
ARG DOCKER_USER=papillon
ARG DOCKER_PASSWORD=root

# Install requirements
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    # pyenv requirements: https://github.com/pyenv/pyenv/wiki#suggested-build-environment
    build-essential \
    curl \
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
    # ca-certificates: git
    # libgl1: OpenCV
    ca-certificates \
    cmake \
    git \
    language-pack-ja \
    #libgl1 \
    sudo

# Install SSH server
RUN apt-get install -y --no-install-recommends openssh-server && \
    sed -i "s/#PubkeyAuthentication yes/PubkeyAuthentication yes/" /etc/ssh/sshd_config

# Install nodejs for JupyterLab extension
RUN curl -sL https://deb.nodesource.com/setup_current.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    rm -rf /var/lib/apt/lists/*

# Create user
RUN useradd -m -u ${DOCKER_UID} -G sudo ${DOCKER_USER} -s /bin/bash && \
    echo ${DOCKER_USER}:${DOCKER_PASSWORD} | chpasswd && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER $DOCKER_UID
ENV HOME /home/${DOCKER_USER}
WORKDIR ${HOME}

# Install pyenv and python 3
ARG PYTHON_VERSION="3.9.10"
ENV PATH "${HOME}/.pyenv/bin:${PATH}"
RUN git clone https://github.com/pyenv/pyenv.git ${HOME}/.pyenv && \
    pyenv install $PYTHON_VERSION && \
    pyenv global $PYTHON_VERSION

# Install JupyterLab
COPY --chown=${DOCKER_USER}:${DOCKER_USER} requirements.txt ./
RUN eval "$(pyenv init --path)" && \
    cat requirements.txt | xargs -n 1 pip install -U --no-cache-dir -f https://download.pytorch.org/whl/cu113/torch_stable.html && \
    rm requirements.txt

COPY --chown=${DOCKER_USER}:${DOCKER_USER} id_rsa.pub ${HOME}/.ssh/
RUN mv ${HOME}/.ssh/id_rsa.pub ${HOME}/.ssh/authorized_keys && \
    chmod 700 ${HOME}/.ssh && \
    chmod 600 ${HOME}/.ssh/authorized_keys

COPY --chown=${DOCKER_USER}:${DOCKER_USER} .bashrc ${HOME}/
COPY --chown=${DOCKER_USER}:${DOCKER_USER} .jupyter/ ${HOME}/.jupyter/

EXPOSE 22 8888 6006
CMD eval "$(pyenv init --path)" && sudo service ssh start && jupyter lab
