# pull official base image
FROM nginx:latest


# 필수 패키지 설치
RUN rm /etc/nginx/conf.d/default.conf
COPY nginx.conf /etc/nginx/conf.d