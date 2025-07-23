from django.conf import settings
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase
from django.urls import reverse
import json


class MatchingAssignmentAppTests(TestCase):

    def test_matching_assignment_introduction_view_loads_successfully(self):
        """Matching Assignment 데모 케이스 설명 페이지가 정상적으로 로드되는지 테스트합니다."""
        url = reverse('matching_assignment_app:matching_assignment_introduction')
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'matching_assignment_app/matching_assignment_introduction.html')
        self.assertContains(response, 'Transportation Assignment')
        self.assertContains(response, 'Resource-Skill Matching')
        self.assertContains(response, 'LCD TFT-CF Matching')

    def test_transport_assignment_introduction_view_loads_successfully(self):
        """Transport Assignment 설명 페이지가 정상적으로 로드되는지 테스트합니다."""
        url = reverse('matching_assignment_app:transport_assignment_introduction')
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'matching_assignment_app/transport_assignment_introduction.html')

    def test_transport_assignment_demo_view_loads_successfully(self):
        """Transport Assignment demo 페이지가 정상적으로 로드되는지 테스트합니다."""
        url = reverse('matching_assignment_app:transport_assignment_demo')
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'matching_assignment_app/transport_assignment_demo.html')
        self.assertContains(response, 'Transportation Assignment Demo')

    def test_transport_assignment_demo_post_request_returns_solution(self):
        """Transport Assignment 데모가 POST 요청 시 최적 생산 계획을 계산하는지 테스트합니다."""
        url = reverse('matching_assignment_app:transport_assignment_demo')

        post_data = {
            'problem_type': 'transport assignment',
            'num_items': '3',
            'zone_name_0': '강남구',
            'zone_name_1': '서초구',
            'zone_name_2': '송파구',
            'driver_name_0': '김기사',
            'cost_0_0': '47', 'cost_0_1': '70', 'cost_0_2': '30',
            'driver_name_1': '이배달',
            'cost_1_0': '42', 'cost_1_1': '58', 'cost_1_2': '23',
            'driver_name_2': '박운송',
            'cost_2_0': '89', 'cost_2_1': '32', 'cost_2_2': '92',
        }

        # POST 요청 시뮬레이션
        response = self.client.post(url, post_data)

        # 결과 검증
        self.assertEqual(response.status_code, 200)
        if settings.SAVE_DATA_FILE:
            self.assertIn("json'으로 서버에 저장되었습니다.", response.context.get('success_save_message', ''))
        self.assertIsNotNone(response.context.get('results'))
        self.assertIn('최적 할당 완료', response.context.get('success_message', ''))
        self.assertContains(response, "결과 요약")
        self.assertContains(response, "최소 총 비용")

    def test_resource_skill_matching_introduction_view_loads_successfully(self):
        """Resource Skill Matching 설명 페이지가 정상적으로 로드되는지 테스트합니다."""
        url = reverse('matching_assignment_app:resource_skill_matching_introduction')
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'matching_assignment_app/resource_skill_matching_introduction.html')

    def test_resource_skill_matching_demo_view_loads_successfully(self):
        """Resource Skill Matching demo 페이지가 정상적으로 로드되는지 테스트합니다."""
        url = reverse('matching_assignment_app:resource_skill_matching_demo')
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'matching_assignment_app/resource_skill_matching_demo.html')
        self.assertContains(response, 'Resource-Skill Matching Demo')

    def test_resource_skill_matching_demo_post_request_returns_solution(self):
        """Resource Skill Matching 데모가 POST 요청 시 최적 매칭을 계산하는지 테스트합니다."""
        url = reverse('matching_assignment_app:resource_skill_matching_demo')

        post_data = {
            'problem_type': 'resource skill',
            'num_projects': '3',
            'selected_resources': ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8'],
            'proj_0_id': 'P1', 'proj_0_name': 'AI 모델 개발', 'proj_0_required_skills': 'Python,ML,SQL',
            'proj_1_id': 'P2', 'proj_1_name': '데이터베이스 마이그레이션', 'proj_1_required_skills': 'AWS,SQL,Cloud',
            'proj_2_id': 'P3', 'proj_2_name': '웹 서비스 프론트엔드', 'proj_2_required_skills': 'React,JavaScript'
        }

        # POST 요청 시뮬레이션
        response = self.client.post(url, post_data)

        # 결과 검증
        self.assertEqual(response.status_code, 200)
        if settings.SAVE_DATA_FILE:
            self.assertIn("json'으로 서버에 저장되었습니다.", response.context.get('success_save_message', ''))
        self.assertIsNotNone(response.context.get('results'))
        self.assertIn('최적 팀 구성 완료', response.context.get('success_message', ''))
        self.assertContains(response, "결과 요약")
        self.assertContains(response, "팀 구성")

    def test_resource_skill_matching_demo_post_request_infeasible_returns(self):
        """Resource Skill Matching 데모가 POST 요청 시 리소스가 부족하여 해결 불능한 경우, 올바른 오류 메시지를 반환하는지 테스트"""
        url = reverse('matching_assignment_app:resource_skill_matching_demo')

        post_data = {
            'problem_type': 'resource skill',
            'num_projects': '3',
            'selected_resources': ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7'],
            'proj_0_id': 'P1', 'proj_0_name': 'AI 모델 개발', 'proj_0_required_skills': 'Python,ML,SQL',
            'proj_1_id': 'P2', 'proj_1_name': '데이터베이스 마이그레이션', 'proj_1_required_skills': 'AWS,SQL,Cloud',
            'proj_2_id': 'P3', 'proj_2_name': '웹 서비스 프론트엔드', 'proj_2_required_skills': 'React,JavaScript'
        }

        # POST 요청 시뮬레이션
        response = self.client.post(url, post_data)

        # 결과 검증
        self.assertEqual(response.status_code, 200)
        if settings.SAVE_DATA_FILE:
            self.assertIn("json'으로 서버에 저장되었습니다.", response.context.get('success_save_message', ''))
        self.assertIsNone(response.context.get('results'))
        self.assertIsNotNone(response.context.get('error_message'))
        self.assertContains(response, "실행 불가능한 문제입니다. 다음 요구사항을 충족할 수 없습니다")

    def test_lcd_cf_tft_introduction_view_loads_successfully(self):
        """LCD TFT-CF Matching 설명 페이지가 정상적으로 로드되는지 테스트합니다."""
        url = reverse('matching_assignment_app:lcd_cf_tft_introduction')
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'matching_assignment_app/lcd_cf_tft_introduction.html')

    def test_lcd_cf_tft_data_generation_view_loads_successfully(self):
        """LCD TFT-CF Matching 데이터 생성 페이지가 정상적으로 로드되는지 테스트합니다."""
        url = reverse('matching_assignment_app:lcd_cf_tft_data_generation')
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'matching_assignment_app/lcd_cf_tft_data_generation.html')

    def test_lcd_cf_tft_data_generation_post_request_generates_data_and_shows_linky(self):
        """데이터 생성 요청(POST) 시, JSON 데이터와 다음 페이지로 넘어가는 버튼이 표시되는지 테스트합니다."""
        url = reverse('matching_assignment_app:lcd_cf_tft_data_generation')

        # 폼에서 전송될 테스트 데이터 구성
        post_data = {
            'num_cf_panels': '4',
            'num_tft_panels': '4',
            'panel_rows': '3',
            'panel_cols': '4',
            'defect_rate': '30',
        }

        # POST 요청 시뮬레이션
        response = self.client.post(url, post_data)

        # 1. 요청 성공 및 올바른 템플릿 사용 확인
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'matching_assignment_app/lcd_cf_tft_data_generation.html')

        # 1. context에 생성된 데이터가 포함되어 있는지 확인
        self.assertIn('generated_data', response.context)
        self.assertIn('generated_data_json_pretty', response.context)
        self.assertIsNotNone(response.context['generated_data'])

        # 3. HTML에 결과 섹션이 렌더링 되었는지 확인
        self.assertContains(response, "Generated Data Preview")
        self.assertContains(response, "Proceed to Small-scale Test with this data")

        # 4. JSON 데이터 미리보기에 핵심 키(예: 'cf_panels')가 포함되어 있는지 확인
        self.assertContains(response, "cf_panels")

        # 5. 생성된 JSON 데이터의 구조와 값이 유효한지 확인
        try:
            generated_data_json = response.context['generated_data_json_pretty']
            generated_data = json.loads(generated_data_json)

            # 설정값이 올바르게 반영되었는지 확인
            settings = generated_data.get('settings', {})
            self.assertEqual(settings.get('num_cf_panels'), 4)
            self.assertEqual(settings.get('num_tft_panels'), 4)
            self.assertEqual(settings.get('panel_rows'), 3)
            self.assertEqual(settings.get('panel_cols'), 4)

            # CF 패널 데이터의 구조가 올바른지 확인
            self.assertEqual(len(generated_data.get('cf_panels', [])), 4)
            self.assertEqual(len(generated_data['cf_panels'][0]['defect_map']), 3)  # Rows
            self.assertEqual(len(generated_data['cf_panels'][0]['defect_map'][0]), 4)  # Cols

        except json.JSONDecodeError:
            self.fail("생성된 데이터가 유효한 JSON 형식이 아닙니다.")


class LcdCfTftSmallScaleDemoTests(TestCase):

    def setUp(self):
        """테스트에 사용할 공통 데이터를 설정합니다."""
        self.test_data = {
            "cf_panels": [
                {"id": "CF1", "rows": 2, "cols": 2, "defect_map": [[0, 1], [0, 0]]},
                {"id": "CF2", "rows": 2, "cols": 2, "defect_map": [[0, 0], [1, 0]]}
            ],
            "tft_panels": [
                {"id": "TFT1", "rows": 2, "cols": 2, "defect_map": [[1, 0], [0, 0]]},
                {"id": "TFT2", "rows": 2, "cols": 2, "defect_map": [[0, 0], [0, 1]]}
            ]
        }
        self.test_data_json_str = json.dumps(self.test_data)
        self.url = reverse('matching_assignment_app:lcd_cf_tft_small_scale_demo')

    def test_view_loads_successfully(self):
        """LCD TFT-CF Matching small demo 페이지가 정상적으로 로드되는지 테스트합니다."""
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'matching_assignment_app/lcd_cf_tft_small_scale_demo.html')
        self.assertContains(response, 'Matching Model: Small-scale Test')

    def test_direct_post_request_returns_solution(self):
        """사용자가 직접 JSON 데이터를 입력하고 POST 요청 시 최적 매칭 결과를 반환하는지 테스트합니다."""

        post_data = {'test_data_json': self.test_data_json_str}
        response = self.client.post(self.url, post_data)

        # 결과 검증
        self.assertEqual(response.status_code, 200)
        self.assertIsNotNone(response.context.get('matching_pairs'))
        self.assertIn('Matching successful', response.context.get('success_message', ''))
        self.assertContains(response, "Optimal Matching Pairs")
        # context의 'submitted_json_data'가 POST로 보낸 데이터와 일치하는지 확인 (입력값 유지 기능 검증)
        self.assertEqual(response.context.get('submitted_json_data'), self.test_data_json_str)

    def test_post_request_with_data_from_session(self):
        """세션을 통해 데이터 생성 페이지로부터 데이터를 받아올 때 정상적으로 처리되는지 테스트합니다."""
        # 1. 세션에 데이터 미리 설정 (데이터 생성 페이지에서 넘어왔다고 가정)
        session = self.client.session
        session['generated_lcd_data'] = self.test_data_json_str
        session.save()

        # 2. 세션 데이터를 가지고 GET 요청으로 페이지에 접속
        response = self.client.get(self.url)

        # 3. 결과 검증 (HTML 렌더링 결과 대신 context 변수를 직접 확인)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context['submitted_json_data'], self.test_data_json_str)

    def test_post_request_with_invalid_json_returns_error(self):
        """잘못된 형식의 JSON 데이터를 제출했을 때 오류 메시지를 반환하는지 테스트합니다."""
        invalid_json_str = '{"cf_panels": [ ...'  # 닫는 괄호가 없는 잘못된 JSON

        post_data = {
            'test_data_json': invalid_json_str
        }

        response = self.client.post(self.url, post_data)

        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.context.get('results'))  # 결과는 없어야 함
        self.assertIsNotNone(response.context.get('error_message'))
        self.assertIn('잘못된 JSON 형식입니다', response.context.get('error_message', ''))

    def test_post_request_returns_solution(self):
        """유효한 JSON 데이터를 POST로 제출했을 때 최적 매칭 결과를 반환하는지 테스트합니다."""
        # POST 폼 데이터 구성
        post_data = {
            'test_data_json': self.test_data_json_str
        }

        # POST 요청 시뮬레이션
        response = self.client.post(self.url, post_data)

        # 결과 검증
        self.assertEqual(response.status_code, 200)
        self.assertIsNotNone(response.context.get('matching_pairs'))
        self.assertIsNotNone(response.context.get('total_yield'))
        self.assertIn('Matching successful', response.context.get('success_message', ''))
        self.assertContains(response, "Optimal Matching Pairs")


class LcdCfTftLargeDemoTests(TestCase):
    def setUp(self):
        self.url = reverse('matching_assignment_app:lcd_cf_tft_large_scale_demo')

    def test_view_loads_successfully(self):
        """LCD TFT-CF Matching large demo 페이지가  GET 요청 시 정상적으로 로드되는지 테스트합니다."""
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'matching_assignment_app/lcd_cf_tft_large_scale_demo.html')
        self.assertContains(response, 'Matching Model: Large-scale Test')

    def test_post_with_make_json_returns_solution(self):
        """'Make JSON Data' 옵션으로 POST 요청 시, 데이터 생성 및 매칭이 성공하는지 테스트합니다."""

        # 'Make JSON Data' 옵션 선택 및 파라미터 설정
        # (테스트 시간을 줄이기 위해 기본값보다 작은 수 사용)
        post_data = {
            'large_data_input_type': 'make_json',
            'num_cf_panels': '100',
            'num_tft_panels': '100',
            'panel_rows': '4',
            'panel_cols': '4',
            'defect_rate': '15',
        }

        # POST 요청 시뮬레이션
        response = self.client.post(self.url, post_data)

        # 결과 검증
        self.assertEqual(response.status_code, 200)

        # 에러 메시지가 없는지 확인
        self.assertIsNone(response.context.get('error_message'))

        # 결과 데이터가 context에 포함되었는지 확인
        self.assertIsNotNone(response.context.get('large_scale_results'))

        # 성공 메시지에 '대규모 매칭 완료' 문구가 포함되었는지 확인
        self.assertIn('대규모 매칭 완료', response.context.get('success_message', ''))

        # 결과 요약 정보가 HTML에 렌더링되었는지 확인
        self.assertContains(response, "Results Overview")
        self.assertContains(response, "Total Yield")
        self.assertContains(response, "Sample Matching Pairs (First 10)")

        # 결과 데이터의 구조 확인
        results = response.context.get('large_scale_results', {})
        self.assertEqual(results.get('num_cf'), 100)
        self.assertEqual(results.get('num_tft'), 100)
        self.assertIn('num_matches', results)

    def test_post_with_upload_json_returns_solution(self):
        """'Upload JSON File' 옵션으로 POST 요청 시, 매칭이 성공하는지 테스트합니다."""
        url = reverse('matching_assignment_app:lcd_cf_tft_large_scale_demo')

        # 테스트용 JSON 데이터 생성
        test_data = {
            "cf_panels": [{"id": f"CF{i}", "rows": 2, "cols": 2, "defect_map": [[0, 1], [0, 0]]} for i in range(5)],
            "tft_panels": [{"id": f"TFT{i}", "rows": 2, "cols": 2, "defect_map": [[1, 0], [0, 0]]} for i in range(5)]
        }
        json_str = json.dumps(test_data)

        # SimpleUploadedFile을 사용하여 메모리 상에서 가상 파일 생성
        uploaded_file = SimpleUploadedFile(
            "test_data.json",
            json_str.encode('utf-8'),
            content_type="application/json"
        )

        post_data = {
            'large_data_input_type': 'upload_json',
            'data_file': uploaded_file,
        }

        response = self.client.post(url, post_data)

        self.assertEqual(response.status_code, 200)
        self.assertIsNotNone(response.context.get('large_scale_results'))
        self.assertIn('대규모 매칭 완료', response.context.get('success_message', ''))
        self.assertEqual(response.context['large_scale_results']['num_cf'], 5)
