ARG BASE_IMAGE="ubuntu:18.04"
FROM ${BASE_IMAGE}

LABEL "maintainer"="lanpn <phan.ngoclan58@gmail.com>"

# System dependencies
RUN apt-get update && apt-get install -y git\
    python3-dev python3-pip libgtk2.0-dev

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN pip3 install --upgrade pip
ARG TF_VERSION="tensorflow==1.15.0"
RUN pip3 install ${TF_VERSION}

WORKDIR /app
RUN pip3 install --upgrade pip
COPY requirements.txt /app/
RUN pip3 install -r requirements.txt

ARG VDIMG_VERSION="v0.1.1"
RUN pip3 install git+https://bitbucket.org/vdsenseplatform/vdimg.git@${VDIMG_VERSION}

COPY . /app
CMD python3 server.py
