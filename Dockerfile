FROM --platform=linux/amd64 python:3.7-slim-bullseye

RUN \
  apt-get -y update && \
  apt-get --fix-broken -y install && \
  apt-get -y install --no-install-recommends \
    build-essential \
    curl \
    gcc \
    git \
    git-lfs \
    gnupg \
    less \
    libopenblas0 \
    screen \
    tmux \
    vim \
    zip

RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | \
  tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | \
  apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && apt-get update -y && apt-get install google-cloud-cli -y

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /covid-data-model

COPY requirements.txt setup.py ./
RUN pip install -r requirements.txt

COPY phc-requirements.txt ./
RUN pip install -r phc-requirements.txt

COPY . .
RUN pip install -e .
