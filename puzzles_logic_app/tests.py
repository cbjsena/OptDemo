from django.conf import settings
from django.test import TestCase
from django.urls import reverse


class PuzzlesLogicAppTests(TestCase):


    def test_puzzles_logic_introduction_view_loads_successfully(self):
        """Puzzles Logic 데모 케이스 설명 페이지가 정상적으로 로드되는지 테스트합니다."""
        url = reverse('puzzles_logic_app:puzzles_logic_introduction')
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'puzzles_logic_app/puzzles_logic_introduction.html')
        self.assertContains(response, '다이어트 문제 (The Diet Problem)')
        self.assertContains(response, '스포츠 스케줄링 (Sports Scheduling)')
        self.assertContains(response, '외판원 문제 (Traveling Salesman Problem, TSP)')
        self.assertContains(response, '스도쿠 해결사 (Sudoku Solver)')

    def test_diet_problem_introduction_view_loads_successfully(self):
        """Diet Problem 설명 페이지가 정상적으로 로드되는지 테스트합니다."""
        url = reverse('puzzles_logic_app:diet_problem_introduction')
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'puzzles_logic_app/diet_problem_introduction.html')


    def test_diet_problem_demo_view_loads_successfully(self):
        """Diet Problem demo 페이지가 정상적으로 로드되는지 테스트합니다."""
        url = reverse('puzzles_logic_app:diet_problem_demo')
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'puzzles_logic_app/diet_problem_demo.html')
        self.assertContains(response, 'The Diet Problem Demo')


    def test_diet_problem_demo_post_request_returns_solution(self):
        """Diet Problem 데모가 POST 요청 시 최적 식단을 계산하는지 테스트합니다."""
        url = reverse('puzzles_logic_app:diet_problem_demo')

        # 1. 각 데모의 form에 맞는 POST 데이터 구성
        # 예: Diet Problem의 경우, 음식 체크박스와 영양소 범위를 전송
        post_data = {
            'problem_type': 'diet_problem',
            'action': 'optimize',
            'num_foods': '4',
            'num_nutrients': '2',
            'nutrient_0_name': '칼로리(kcal)',
            'nutrient_0_min': '100',
            'nutrient_0_max': '2500',
            'nutrient_1_name': '단백질g)',
            'nutrient_1_min': '20',
            'nutrient_1_max': '100',
            'food_0_name': '우유(100ml)',
            'food_0_cost': '150',
            'food_0_min_intake': '2',
            'food_0_max_intake': '10',
            'nutrient_val_0_0': '60',
            'nutrient_val_0_1': '3.2',
            'food_1_name': '계란(1개)',
            'food_1_cost': '300',
            'food_1_min_intake': '0',
            'food_1_max_intake': '5',
            'nutrient_val_1_0': '80',
            'nutrient_val_1_1': '6',
            'food_2_name': '식빵(1장)',
            'food_2_cost': '200',
            'food_2_min_intake': '0',
            'food_2_max_intake': '10',
            'nutrient_val_2_0': '70',
            'nutrient_val_2_1': '2.5',
            'food_3_name': '닭가슴살(100g)',
            'food_3_cost': '1500',
            'food_3_min_intake': '2',
            'food_3_max_intake': '5',
            'nutrient_val_3_0': '110',
            'nutrient_val_3_1': '23'
        }

        # 2. POST 요청 시뮬레이션
        response = self.client.post(url, post_data)

        # 3. 결과 검증
        self.assertEqual(response.status_code, 200)
        if settings.SAVE_DATA_FILE:
            self.assertIn("json'으로 서버에 저장되었습니다.", response.context.get('success_save_message', ''))
        self.assertIsNotNone(response.context.get('results'))  # 결과 데이터가 있는지 확인
        self.assertIn('최적 식단 계산 완료! 최소 비용', response.context.get('success_message', ''))  # 성공 메시지 확인
        self.assertContains(response, '최소 총 비용')  # 결과 페이지에 특정 텍스트가 있는지 확인


    def test_sports_scheduling_introduction_view_loads_successfully(self):
        """Sports Scheduling 설명 페이지가 정상적으로 로드되는지 테스트합니다."""
        url = reverse('puzzles_logic_app:sports_scheduling_introduction')
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'puzzles_logic_app/sports_scheduling_introduction.html')


    def test_sports_scheduling_demo_view_loads_successfully(self):
        """Sports Scheduling demo 페이지가 정상적으로 로드되는지 테스트합니다."""
        url = reverse('puzzles_logic_app:sports_scheduling_demo')
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'puzzles_logic_app/sports_scheduling_demo.html')
        self.assertContains(response, 'Sports Scheduling Demo')

    def test_sports_scheduling_demo_post_request_returns_solution(self):
        """Sports Scheduling 데모가 POST 요청 시 최적의 시즌 대진표를 생성하는지 테스트합니다."""
        url = reverse('puzzles_logic_app:sports_scheduling_demo')

        # 1. 각 데모의 form에 맞는 POST 데이터 구성
        post_data = {
            'problem_type': 'sports_scheduling',
            'num_teams': '4',
            'schedule_type': 'double',
            'objective_choice': 'minimize_travel',
            'max_consecutive': '3',
            'solver_type': 'ORTOOLS',  # 로그의 값은 리스트 형태의 문자열이므로, 실제 전송되는 단일 값 'ORTOOLS'로 수정
            'team_0_name': '한화',
            'team_1_name': 'LG',
            'team_2_name': '롯데',
            'team_3_name': 'KIA',
        }

        # 2. POST 요청 시뮬레이션
        response = self.client.post(url, post_data)

        # 3. 결과 검증
        self.assertEqual(response.status_code, 200)
        if settings.SAVE_DATA_FILE:
            self.assertIn("json'으로 서버에 저장되었습니다.", response.context.get('success_save_message', ''))
        self.assertIsNotNone(response.context.get('results'))  # 결과 데이터가 있는지 확인
        self.assertIn('Total distance', response.context.get('success_message', ''))  # 성공 메시지 확인
        self.assertContains(response, '시즌 대진표 요약')  # 결과 페이지에 특정 텍스트가 있는지 확인


    def test_tsp_introduction_view_loads_successfully(self):
        """TSP 설명 페이지가 정상적으로 로드되는지 테스트합니다."""
        url = reverse('puzzles_logic_app:tsp_introduction')
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'puzzles_logic_app/tsp_introduction.html')


    def test_tsp_demo_view_loads_successfully(self):
        """TSP 설명 페이지가 정상적으로 로드되는지 테스트합니다."""
        url = reverse('puzzles_logic_app:tsp_demo')
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'puzzles_logic_app/tsp_demo.html')
        self.assertContains(response, 'Traveling Salesperson Problem (TSP) Demo')


    def test_tsp_demo_post_request_returns_solution(self):
        """TSP 데모가 POST 요청 시 최단 경로를 계산하는지 테스트합니다."""
        url = reverse('puzzles_logic_app:tsp_demo')

        # name이 'cities'인 여러 개의 값을 리스트로 전달
        post_data = {
            'cities': ['서울', '부산', '광주', '대전', '강릉']
        }

        # POST 요청 시뮬레이션
        response = self.client.post(url, post_data)

        # 결과 검증
        self.assertEqual(response.status_code, 200)
        if settings.SAVE_DATA_FILE:
            self.assertIn("json'으로 서버에 저장되었습니다.", response.context.get('success_save_message', ''))
        self.assertIsNotNone(response.context.get('results'))
        self.assertIn('최단 경로를 찾았습니다', response.context.get('success_message', ''))
        self.assertContains(response, "최적 경로")  # 결과 페이지에 특정 텍스트가 있는지 확인

        # 결과 경로에 출발 도시(서울)가 포함되어 있는지 확인
        results = response.context.get('results', {})
        self.assertIn('서울', results.get('tour_cities', ''))


    def test_sudoku_instruction_view_loads_successfully(self):
        """Sudoku 설명 페이지가 정상적으로 로드되는지 테스트합니다."""
        url = reverse('puzzles_logic_app:sudoku_introduction')
        response = self.client.get(url)

        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'puzzles_logic_app/sudoku_introduction.html')


    def test_sudoku_demo_view_loads_successfully(self):
        """Sudoku 데모 페이지가 GET 요청 시 정상적으로 로드되는지 테스트합니다."""
        url = reverse('puzzles_logic_app:sudoku_demo')
        response = self.client.get(url)

        self.assertEqual(response.status_code, 200)  # 1. HTTP 상태 코드가 200인지 확인
        self.assertTemplateUsed(response, 'puzzles_logic_app/sudoku_demo.html')  # 2. 올바른 템플릿을 사용하는지 확인
        self.assertContains(response, 'Sudoku Demo')  # 3. 페이지 제목이 HTML에 포함되어 있는지 확인


    # ----------------------------------------------------------------
    # 테스트 2: 데모 View의 POST 요청 및 결과 검증
    # ----------------------------------------------------------------
    def test_sudoku_demo_post_request_returns_solution(self):
        """Sudoku 데모 페이지가 POST 요청 시 퍼즐을 해결하고 결과를 반환하는지 테스트합니다."""
        url = reverse('puzzles_logic_app:sudoku_demo')

        # POST로 전송할 폼 데이터 생성 (기본값 또는 간단한 케이스)
        # 9x9 'easy' 퍼즐을 풀도록 요청
        post_data = {
            'size': '9',
            'difficulty': 'easy',
        }
        # 초기 퍼즐 상태를 form data에 추가 (실제로는 view가 GET에서 생성하지만, POST 테스트에서는 직접 구성)
        # 여기서는 간단히 하기 위해 빈 퍼즐을 보낸다고 가정. View 로직에 따라 필요한 데이터를 채워야 함.
        for i in range(9):
            for j in range(9):
                post_data[f'cell_{i}_{j}'] = '0'

        # 테스트 클라이언트로 POST 요청 전송
        response = self.client.post(url, post_data)

        self.assertEqual(response.status_code, 200)  # 1. 요청이 성공했는지 확인
        if settings.SAVE_DATA_FILE:
            self.assertIn("json'으로 서버에 저장되었습니다.", response.context.get('success_save_message', ''))
        self.assertIsNotNone(response.context.get('results'))  # 2. 결과 데이터가 context에 있는지 확인
        self.assertIn('스도쿠 퍼즐을 성공적으로 풀었습니다', response.context.get('success_message', ''))  # 3. 성공 메시지가 있는지 확인
        self.assertContains(response, 'Solver Result')  # 4. 결과 섹션이 HTML에 렌더링 되었는지 확인