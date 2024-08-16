FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04
 
LABEL maintainer="friedrich.dang@setlabs.de"

ARG USERNAME=vscode

ARG USER_UID=1004

ARG USER_GID=$USER_UID
 
#####################

# Create a non-root user to use if preferred - see https://aka.ms/vscode-remote/containers/non-root-user.

#####################

RUN apt-get update \
    && groupadd --gid $USER_GID $USERNAME \
    && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
    # [Optional] Add sudo support for non-root user
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME  \
    # Clean up
    && apt-get autoremove -y \
    && apt-get clean -y

# Set CUDA-related environment variables
ARG TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6+PTX"
ARG FORCE_CUDA="1"
ENV CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
 
RUN apt-get -y update
 
RUN apt-get update && apt install -y python3 python3-pip apt-transport-https ca-certificates gnupg software-properties-common wget vim lsb-release zip unzip curl git ninja-build libboost-dev build-essential ffmpeg libsm6 libxext6 wget vim nano
 
RUN pip3 install --upgrade pip
 
COPY requirements.txt .
 
RUN pip3 install -r requirements.txt
 
WORKDIR /workspace/CenterPoint

COPY . /workspace/CenterPoint 

RUN /bin/bash setup.sh

ENV PYTHONPATH="${PYTHONPATH}:/workspace/CenterPoint"
 