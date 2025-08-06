from django.conf import settings
from django.test import TestCase
from django.urls import reverse


class ResourceAllocationAppTests(TestCase):

    def test_resource_allocation_introduction_view_loads_successfully(self):
        """Resource Allocation 데모 케이스 설명 페이지가 정상적으로 로드되는지 테스트합니다."""
        url = reverse('resource_allocation_app:resource_allocation_introduction')
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'resource_allocation_app/resource_allocation_introduction.html')
        self.assertContains(response, 'Budget Allocation')
        self.assertContains(response, 'Data Center Capacity')
        self.assertContains(response, 'Nurse Rostering')

    def test_budget_allocation_introduction_view_loads_successfully(self):
        """Budget Allocation 설명 페이지가 GET 요청 시 정상적으로 로드되는지 테스트합니다."""
        url = reverse('resource_allocation_app:budget_allocation_introduction')
        response = self.client.get(url)

        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'resource_allocation_app/budget_allocation_introduction.html')
        self.assertContains(response, '예산 분배 최적화')

    def test_budget_allocation_demo_view_loads_successfully(self):
        """Budget Allocation 데모 페이지가 GET 요청 시 정상적으로 로드되는지 테스트합니다."""
        url = reverse('resource_allocation_app:budget_allocation_demo')
        response = self.client.get(url)

        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'resource_allocation_app/budget_allocation_demo.html')
        self.assertContains(response, 'Budget Allocation Optimization Demo')

    def test_budget_allocation_demo_post_request_returns_solution(self):
        """Budget Allocation 데모가 POST 요청 시 최적 예산을 계산하는지 테스트합니다."""
        url = reverse('resource_allocation_app:budget_allocation_demo')

        # 데모 페이지의 기본값과 유사한 POST 데이터 구성
        # HTML의 form 내 input들의 name 속성을 기반으로 작성합니다.
        post_data = {
            'problem_type': 'budget',
            'num_items': '3',
            'total_budget': '1000',

            # Item 0
            'item_0_name': 'item_0',
            'item_0_return_coefficient': '1.1',
            'item_0_min_alloc': '0',
            'item_0_max_alloc': '1000',

            # Item 1
            'item_1_name': 'item_1',
            'item_1_return_coefficient': '3.1',
            'item_1_min_alloc': '0',
            'item_1_max_alloc': '200',

            # Item 2
            'item_2_name': 'item_2',
            'item_2_return_coefficient': '2.1',
            'item_2_min_alloc': '0',
            'item_2_max_alloc': '300',
        }

        # POST 요청 시뮬레이션
        response = self.client.post(url, post_data)

        # 결과 검증
        self.assertEqual(response.status_code, 200)
        if settings.SAVE_DATA_FILE:
            self.assertIn("json'으로 서버에 저장되었습니다.", response.context.get('success_save_message', ''))
        # 성공 메시지나 결과가 context에 포함되어 있는지 확인
        self.assertIsNotNone(response.context.get('results'))
        self.assertIn('최적 예산 분배 수립 완료', response.context.get('success_message', ''))
        # 결과 페이지에 특정 텍스트가 렌더링 되었는지 확인
        self.assertContains(response, "결과 요약")
        self.assertContains(response, "계산된 최대 기대 수익")

    def test_data_center_capacity_introduction_view_loads_successfully(self):
        """Datacenter Capacity 설명 페이지가 GET 요청 시 정상적으로 로드되는지 테스트합니다."""
        url = reverse('resource_allocation_app:data_center_capacity_introduction')
        response = self.client.get(url)

        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'resource_allocation_app/data_center_capacity_introduction.html')
        self.assertContains(response, '데이터 센터 용량 계획')

    def test_data_center_capacity_demo_view_loads_successfully(self):
        """Datacenter Capacity 데모 페이지가 GET 요청 시 정상적으로 로드되는지 테스트합니다."""
        url = reverse('resource_allocation_app:data_center_capacity_demo')
        response = self.client.get(url)

        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'resource_allocation_app/data_center_capacity_demo.html')
        self.assertContains(response, 'Data Center Capacity Planning Demo')

    def test_data_center_capacity_demo_post_request_returns_solution(self):
        """Data Center Capacity 데모가 POST 요청 시 최적 계획을 계산하는지 테스트합니다."""
        url = reverse('resource_allocation_app:data_center_capacity_demo')

        post_data = {
            'problem_type': 'datacenter',
            'num_server_types': '2',
            'num_services': '2',
            'total_budget': '100000',
            'total_power_kva': '50',
            'total_space_sqm': '10',
            # 서버 유형 1 정보
            'server_0_id': 'SrvA',
            'server_0_cost': '500',
            'server_0_cpu_cores': '48',
            'server_0_ram_gb': '256',
            'server_0_storage_tb': '10',
            'server_0_power_kva': '0.5',
            'server_0_space_sqm': '0.2',
            # 서버 유형 2 정보
            'server_1_id': 'SrvB',
            'server_1_cost': '300',
            'server_1_cpu_cores': '32',
            'server_1_ram_gb': '128',
            'server_1_storage_tb': '5',
            'server_1_power_kva': '0.3',
            'server_1_space_sqm': '0.1',
            # 서비스 1 정보
            'service_0_id': 'WebPool',
            'service_0_revenue_per_unit': '100',
            'service_0_req_cpu_cores': '4',
            'service_0_req_ram_gb': '8',
            'service_0_req_storage_tb': '0.1',
            'service_0_max_units': '50',
            # 서비스 2 정보
            'service_1_id': 'DBFarm',
            'service_1_revenue_per_unit': '200',
            'service_1_req_cpu_cores': '8',
            'service_1_req_ram_gb': '16',
            'service_1_req_storage_tb': '0.5',
            'service_1_max_units': '20',
        }

        # POST 요청 시뮬레이션
        response = self.client.post(url, post_data)

        # 결과 검증
        self.assertEqual(response.status_code, 200)
        if settings.SAVE_DATA_FILE:
            self.assertIn("json'으로 서버에 저장되었습니다.", response.context.get('success_save_message', ''))
        self.assertIsNotNone(response.context.get('results'))
        self.assertIn('데이터 센터 용량 계획 최적화 완료', response.context.get('success_message', ''))
        self.assertContains(response, "최적화 결과 요약")
        self.assertContains(response, "계산된 총 이익")


class NurseRosteringTests(TestCase):

    def test_nurse_rostering_introduction_view_loads_successfully(self):
        """Nurse Rostering 설명 페이지가 GET 요청 시 정상적으로 로드되는지 테스트합니다."""
        url = reverse('resource_allocation_app:nurse_rostering_introduction')
        response = self.client.get(url)

        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'resource_allocation_app/nurse_rostering_introduction.html')
        self.assertContains(response, 'Nurse Rostering Problem')

    def test_nurse_rostering_demo_view_loads_successfully(self):
        """Nurse Rostering 데모 페이지가 GET 요청 시 정상적으로 로드되는지 테스트합니다."""
        url = reverse('resource_allocation_app:nurse_rostering_demo')
        response = self.client.get(url)

        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'resource_allocation_app/nurse_rostering_demo.html')
        self.assertContains(response, 'Nurse Rostering Demo')

    def test_nurse_rostering_advanced_view_loads_successfully(self):
        """Advanced Nurse Rostering 데모 페이지가 GET 요청 시 정상적으로 로드되는지 테스트합니다."""
        url = reverse('resource_allocation_app:nurse_rostering_advanced_demo')
        response = self.client.get(url)

        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'resource_allocation_app/nurse_rostering_advanced_demo.html')
        self.assertContains(response, 'Advanced Nurse Rostering Demo')

    def test_nurse_rostering_demo_post_request_returns_solution(self):
        """Nurse Rostering 데모가 POST 요청 시 최적 계획을 계산하는지 테스트합니다."""
        url = reverse('resource_allocation_app:nurse_rostering_demo')

        post_data = {
            'problem_type': 'nurse_rostering',
            'num_nurses': '15',
            'num_days': '7',
            'min_shifts': '2',
            'max_shifts': '6',
            'shift_0_req': '4',  # 주간(D) 필요 인원
            'shift_1_req': '3',  # 오후(E) 필요 인원
            'shift_2_req': '2',  # 야간(N) 필요 인원
        }

        # POST 요청 시뮬레이션
        response = self.client.post(url, post_data)

        # 결과 검증
        self.assertEqual(response.status_code, 200)
        if settings.SAVE_DATA_FILE:
            self.assertIn("json'으로 서버에 저장되었습니다.", response.context.get('success_save_message', ''))
        self.assertIsNotNone(response.context.get('results'))
        self.assertIn('최적의 근무표를 생성했습니다!', response.context.get('success_message', ''))
        self.assertContains(response, "생성된 근무표")
        self.assertContains(response, "간호사별 근무일 수 요약")

    def test_nurse_rostering_advanced_post_request_infeasible_returns(self):
        """Advanced Nurse Rostering 데모가 POST 요청 시 해결 불능한 경우, 올바른 오류 메시지를 반환하는지 테스트"""
        url = reverse('resource_allocation_app:nurse_rostering_advanced_demo')

        post_data = {
            'problem_type': 'nurse_rostering',
            'num_nurses': '15',
            'num_days': '7',
            'min_shifts': '2',
            'max_shifts': '6',
            # 간호사 15명 정보
            'nurse_0_name': 'NurA', 'nurse_0_skill': 'L', 'nurse_0_vacation': '',
            'nurse_1_name': 'NurB', 'nurse_1_skill': 'M', 'nurse_1_vacation': '',
            'nurse_2_name': 'NurC', 'nurse_2_skill': 'L', 'nurse_2_vacation': '',
            'nurse_3_name': 'NurD', 'nurse_3_skill': 'H', 'nurse_3_vacation': '',
            'nurse_4_name': 'NurE', 'nurse_4_skill': 'M', 'nurse_4_vacation': '',
            'nurse_5_name': 'NurF', 'nurse_5_skill': 'H', 'nurse_5_vacation': '',
            'nurse_6_name': 'NurG', 'nurse_6_skill': 'M', 'nurse_6_vacation': '',
            'nurse_7_name': 'NurH', 'nurse_7_skill': 'L', 'nurse_7_vacation': '',
            'nurse_8_name': 'NurI', 'nurse_8_skill': 'H', 'nurse_8_vacation': '',
            'nurse_9_name': 'NurJ', 'nurse_9_skill': 'M', 'nurse_9_vacation': '',
            'nurse_10_name': 'NurK', 'nurse_10_skill': 'L', 'nurse_10_vacation': '',
            'nurse_11_name': 'NurL', 'nurse_11_skill': 'M', 'nurse_11_vacation': '',
            'nurse_12_name': 'NurM', 'nurse_12_skill': 'H', 'nurse_12_vacation': '',
            'nurse_13_name': 'NurN', 'nurse_13_skill': 'M', 'nurse_13_vacation': '',
            'nurse_14_name': 'NurO', 'nurse_14_skill': 'M', 'nurse_14_vacation': '',
            # 시프트별 필요인원
            'req_Day_H': '1', 'req_Day_M': '2', 'req_Day_L': '1',
            'req_Aft_H': '1', 'req_Aft_M': '2', 'req_Aft_L': '1',
            'req_Ngt_H': '1', 'req_Ngt_M': '0', 'req_Ngt_L': '1',
            # 공정성 옵션
            'fairness_options': ['fair_weekends', 'fair_nights', 'fair_offs']
        }

        # POST 요청 시뮬레이션
        response = self.client.post(url, post_data)

        # 결과 검증
        self.assertEqual(response.status_code, 200)
        if settings.SAVE_DATA_FILE:
            self.assertIn("json'으로 서버에 저장되었습니다.", response.context.get('success_save_message', ''))
        self.assertIsNone(response.context.get('results'))
        self.assertIn('Optimal solution not found. Solver status: INFEASIBLE', response.context.get('error_message', ''))

    def test_nurse_rostering_advanced_post_request_feasible_returns(self):
        """Advanced Nurse Rostering 데모가 POST 요청 시 최적 계획을 계산하는지 테스트합니다."""
        url = reverse('resource_allocation_app:nurse_rostering_advanced_demo')

        post_data = {
            'problem_type': 'nurse_rostering',
            'num_nurses': '19',
            'num_days': '7',
            'min_shifts': '2',
            'max_shifts': '7',
            'nurse_0_name': 'Nur1', 'nurse_0_skill': 'L', 'nurse_0_vacation': '',
            'nurse_1_name': 'Nur2', 'nurse_1_skill': 'M', 'nurse_1_vacation': '',
            'nurse_2_name': 'Nur3', 'nurse_2_skill': 'L', 'nurse_2_vacation': '',
            'nurse_3_name': 'Nur4', 'nurse_3_skill': 'H', 'nurse_3_vacation': '',
            'nurse_4_name': 'Nur5', 'nurse_4_skill': 'M', 'nurse_4_vacation': '',
            'nurse_5_name': 'Nur6', 'nurse_5_skill': 'H', 'nurse_5_vacation': '',
            'nurse_6_name': 'Nur7', 'nurse_6_skill': 'M', 'nurse_6_vacation': '',
            'nurse_7_name': 'Nur8', 'nurse_7_skill': 'L', 'nurse_7_vacation': '',
            'nurse_8_name': 'Nur9', 'nurse_8_skill': 'H', 'nurse_8_vacation': '',
            'nurse_9_name': 'Nur10', 'nurse_9_skill': 'M', 'nurse_9_vacation': '',
            'nurse_10_name': 'Nur11', 'nurse_10_skill': 'L', 'nurse_10_vacation': '',
            'nurse_11_name': 'Nur12', 'nurse_11_skill': 'M', 'nurse_11_vacation': '',
            'nurse_12_name': 'Nur13', 'nurse_12_skill': 'H', 'nurse_12_vacation': '',
            'nurse_13_name': 'Nur14', 'nurse_13_skill': 'M', 'nurse_13_vacation': '',
            'nurse_14_name': 'Nur15', 'nurse_14_skill': 'M', 'nurse_14_vacation': '',
            'nurse_15_name': 'Nur16', 'nurse_15_skill': 'L', 'nurse_15_vacation': '',
            'nurse_16_name': 'Nur17', 'nurse_16_skill': 'L', 'nurse_16_vacation': '',
            'nurse_17_name': 'Nur18', 'nurse_17_skill': 'M', 'nurse_17_vacation': '',
            'nurse_18_name': 'Nur19', 'nurse_18_skill': 'H', 'nurse_18_vacation': '',
            # 시프트별 필요인원
            'req_Day_H': '1', 'req_Day_M': '2', 'req_Day_L': '1',
            'req_Aft_H': '1', 'req_Aft_M': '2', 'req_Aft_L': '1',
            'req_Ngt_H': '1', 'req_Ngt_M': '0', 'req_Ngt_L': '1',
            # 공정성 옵션
            'fairness_options': ['fair_weekends', 'fair_nights', 'fair_offs']
        }

        # POST 요청 시뮬레이션
        response = self.client.post(url, post_data)

        # 결과 검증
        self.assertEqual(response.status_code, 200)
        if settings.SAVE_DATA_FILE:
            self.assertIn("json'으로 서버에 저장되었습니다.", response.context.get('success_save_message', ''))
        self.assertIsNotNone(response.context.get('results'))
        self.assertIn('최적의 근무표를 생성했습니다', response.context.get('success_message', ''))
        self.assertContains(response, "생성된 근무표")
        self.assertContains(response, "간호사별 근무 통계")