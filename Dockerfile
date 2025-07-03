# pull official base image
FROM python:3.11-slim-buster


# 필수 패키지 설치
RUN apt-get update && \
    apt-get install -y wget build-essential libreadline-dev libsqlite3-dev zlib1g-dev make gcc

# SQLite 최신 버전 다운로드 및 빌드
RUN wget https://www.sqlite.org/2024/sqlite-autoconf-3450100.tar.gz && \
    tar xzf sqlite-autoconf-3450100.tar.gz && \
    cd sqlite-autoconf-3450100 && \
    ./configure && make && make install && \
    cd .. && rm -rf sqlite-autoconf*

# shared library 로드
RUN ldconfig

# set work directory
WORKDIR /usr/src/app

# set enviroment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

COPY . /usr/src/app/

# install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt