FROM --platform=linux/amd64 python:3.7-slim-bullseye as deps-image

RUN \
  apt-get -y update && \
  apt-get --fix-broken -y install && \
  apt-get -y install --no-install-recommends \
    build-essential \
    curl \
    gcc \
    git \
    git-lfs \
    less \
    libopenblas0 \
    screen \
    tmux \
    vim \
    zip

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /covid-data-model

COPY requirements.txt setup.py .

RUN pip install -r requirements.txt
RUN pip install google-cloud-storage fsspec gcsfs python-decouple ipython

COPY . .
RUN pip install .
