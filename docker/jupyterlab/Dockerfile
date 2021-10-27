FROM ubuntu:18.04
LABEL maintainer "pystyle"

ENV LC_ALL C.UTF-8
ENV PATH "/root/.pyenv/bin:${PATH}"
ARG PYTHON_VERSION="3.9.7"

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

# Install nodejs for JupyterLab extension
RUN curl -sL https://deb.nodesource.com/setup_current.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    rm -rf /var/lib/apt/lists/*

# Install pyenv and python 3
RUN git clone https://github.com/pyenv/pyenv.git ~/.pyenv && \
    pyenv install $PYTHON_VERSION && \
    pyenv global $PYTHON_VERSION

# Install JupyterLab
WORKDIR /notebook
COPY requirements.txt .
RUN eval "$(pyenv init --path)" && \
    cat requirements.txt | xargs -n 1 pip install -U --no-cache-dir && \
    jupyter notebook --generate-config -y && \
    rm requirements.txt
COPY jupyter_lab_config.py /root/.jupyter/
# fix issue about black https://github.com/psf/black/issues/1143
RUN mkdir -p /root/.cache/black/21.9b0/

EXPOSE 8888
CMD eval "$(pyenv init --path)" && jupyter lab
