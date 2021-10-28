FROM ubuntu:18.04
LABEL maintainer "pystyle"

ENV LC_ALL C.UTF-8

# Install SSH server
RUN apt-get update && \
    apt-get install -y --no-install-recommends openssh-server && \
    rm -rf /var/lib/apt/lists/* && \
    echo "root:root" | chpasswd && \
    sed -i "s/#PermitRootLogin prohibit-password/PermitRootLogin yes/" /etc/ssh/sshd_config

EXPOSE 22
CMD service ssh start && /bin/bash
