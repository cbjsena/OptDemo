# pull official base image
FROM python:3.11-slim-buster

# set enviroment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# set work directory
WORKDIR /usr/src/app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .
COPY .env.prod .env
# 정적 파일 폴더 생성 및 수집
RUN mkdir -p /usr/src/app/static
RUN python manage.py collectstatic --noinput

# Cloud SQL용 소켓 경로 생성
RUN mkdir -p /cloudsql
ENV CLOUD_SQL_CONNECTION_NAME=my-optimization-demo:asia-northeast3:my-optimization-db

# 포트 8080에 맞춰 gunicorn 실행
CMD ["gunicorn", "optdemo_project.wsgi:application", "--bind", "0.0.0.0:8080", "--log-level=debug"]

