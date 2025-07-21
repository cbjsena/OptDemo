#!/bin/sh

# 이 스크립트는 컨테이너가 시작될 때마다 가장 먼저 실행됩니다.

# 1. 데이터베이스 마이그레이션을 실행합니다. (배포 시 좋은 습관)
echo "Applying database migrations..."
python manage.py migrate --noinput

# 2. 정적 파일을 수집합니다.
# 이 시점에는 docker-compose.yml의 static_volume이 이미 마운트된 상태이므로,
# 이 명령은 비어있는 볼륨에 파일들을 채워 넣게 됩니다.
echo "Collecting static files..."
python manage.py collectstatic --noinput

# 3. Dockerfile의 CMD로 전달된 원래 명령어를 실행합니다.
# 이 예제에서는 gunicorn 서버를 시작하는 명령이 됩니다.
exec "$@"