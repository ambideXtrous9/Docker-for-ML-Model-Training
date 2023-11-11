FROM ubuntu:latest

RUN apt -y update \
    && apt install -y htop wget

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir root/.conda \
    && sh Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN conda create -y -n ml

COPY . /src 


RUN /bin/bash -c "cd src \
    && source activate ml \
    && python -m pip install --upgrade pip \
    && pip install -r requirements.txt"