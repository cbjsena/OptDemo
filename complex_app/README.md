# complex_app

`complex_app`은 고난도 최적화 데모를 위한 Django 앱입니다.

## 포함 기능

- Complex Optimization 소개 페이지
- 3D Palletizing 소개 페이지
- 3D Palletizing 데모 페이지
  - 팔렛 크기/중량 입력
  - 박스 타입(치수, 중량, 수량, 회전 허용) 입력
  - 그리디 3D 배치 결과(좌표/적재율/미적재 목록) 출력

## 핵심 파일

- `complex_app/views.py`
- `complex_app/urls.py`
- `complex_app/solvers/palletizing_solver.py`
- `complex_app/templates/complex_app/*.html`
- `complex_app/tests.py`

