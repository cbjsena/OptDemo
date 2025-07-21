from django.conf import settings
from django.test import TestCase
from django.urls import reverse


class VrpDemoTests(TestCase):
    """VRP(차량 경로 문제) 데모에 대한 테스트"""

    def test_introduction_view_loads_successfully(self):
        """Vehicle Routing Problem 설명 페이지가 GET 요청 시 정상적으로 로드되는지 테스트합니다."""
        url = reverse('routing_logistics_app:vrp_introduction')
        response = self.client.get(url)

        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'routing_logistics_app/vrp_introduction.html')
        self.assertContains(response, '차량 경로 문제 (Vehicle Routing Problem - VRP)')

    def test_demo_view_loads_successfully(self):
        """Vehicle Routing Problem 데모 페이지가 GET 요청 시 정상적으로 로드되는지 테스트합니다."""
        url = reverse('routing_logistics_app:vrp_demo')
        response = self.client.get(url)

        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'routing_logistics_app/vrp_demo.html')
        self.assertContains(response, 'Vehicle Routing Problem (VRP) Demo - Graphical Input')

    def test_demo_post_request_returns_solution(self):
        """VRP 데모가 POST 요청 시 최적 경로를 계산하는지 테스트합니다."""
        url = reverse('routing_logistics_app:vrp_demo')

        post_data = {
            'problem_type': 'vrp',
            'num_customers': '5',
            'num_vehicles': '3',
            'depot_x': '300.0', 'depot_y': '250.0',
            'cust_0_id': 'C1', 'cust_0_x': '103.0', 'cust_0_y': '120.0', 'cust_0_demand': '30',
            'cust_1_id': 'C2', 'cust_1_x': '510.0', 'cust_1_y': '150.0', 'cust_1_demand': '30',
            'cust_2_id': 'C3', 'cust_2_x': '171.0', 'cust_2_y': '317.0', 'cust_2_demand': '40',
            'cust_3_id': 'C4', 'cust_3_x': '486.0', 'cust_3_y': '283.0', 'cust_3_demand': '40',
            'cust_4_id': 'C5', 'cust_4_x': '384.0', 'cust_4_y': '45.0', 'cust_4_demand': '30',
        }

        # POST 요청 시뮬레이션
        response = self.client.post(url, post_data)

        # 결과 검증
        self.assertEqual(response.status_code, 200)
        self.assertIsNotNone(response.context.get('opt_results'))
        self.assertIn('VRP 최적 경로 계산 완료', response.context.get('success_message', ''))
        if settings.SAVE_DATA_FILE:
            self.assertIn("json'으로 서버에 저장되었습니다.", response.context.get('success_save_message', ''))
        self.assertContains(response, "결과 요약")
        self.assertContains(response, "차량별 최적 경로")
        self.assertContains(response, "경로 시각화 (VRP)")

    def test_advanced_view_loads_successfully(self):
        """Vehicle Routing Problem Advanced 페이지가 GET 요청 시 정상적으로 로드되는지 테스트합니다."""
        url = reverse('routing_logistics_app:vrp_advanced')
        response = self.client.get(url)

        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'routing_logistics_app/vrp_advanced.html')
        self.assertContains(response, '차량 경로 최적화 모델 정의')
        self.assertContains(response, '주요 VRP 변형 모델 상세 설명')
        self.assertContains(response, '종합 정리')


class CvrpDemoTests(TestCase):
    """CVRP(용량 제약 차량 경로 문제) 데모에 대한 테스트"""

    def test_introduction_view_loads_successfully(self):
        url = reverse('routing_logistics_app:cvrp_introduction')
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'routing_logistics_app/cvrp_introduction.html')
        self.assertContains(response, '용량 제약 차량 경로 문제 (Capacitated VRP - CVRP)')

    def test_demo_view_loads_successfully(self):
        url = reverse('routing_logistics_app:cvrp_demo')
        response = self.client.get(url)

        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'routing_logistics_app/cvrp_demo.html')
        self.assertContains(response, 'Capacitated VRP (CVRP) Demo - Graphical Input')

    def test_demo_post_request_returns_solution(self):
        """CVRP 데모가 POST 요청 시 용량 제약을 고려한 최적 경로를 계산하는지 테스트합니다."""
        url = reverse('routing_logistics_app:cvrp_demo')

        # CVRP의 특성에 맞는 POST 데이터 (차량 용량 추가)
        post_data = {
            'problem_type': 'cvrp',
            'num_customers': '5',
            'num_vehicles': '3',
            'vehicle_capacity': '100',
            'depot_x': '300.0',
            'depot_y': '250.0',
            'cust_0_id': 'C1', 'cust_0_x': '103.0', 'cust_0_y': '120.0', 'cust_0_demand': '30',
            'cust_1_id': 'C2', 'cust_1_x': '510.0', 'cust_1_y': '150.0', 'cust_1_demand': '30',
            'cust_2_id': 'C3', 'cust_2_x': '171.0', 'cust_2_y': '317.0', 'cust_2_demand': '40',
            'cust_3_id': 'C4', 'cust_3_x': '486.0', 'cust_3_y': '283.0', 'cust_3_demand': '40',
            'cust_4_id': 'C5', 'cust_4_x': '384.0', 'cust_4_y': '45.0', 'cust_4_demand': '30',
        }

        response = self.client.post(url, post_data)

        # 결과 검증
        self.assertEqual(response.status_code, 200)
        if settings.SAVE_DATA_FILE:
            self.assertIn("json'으로 서버에 저장되었습니다.", response.context.get('success_save_message', ''))
        self.assertIsNotNone(response.context.get('opt_results'))
        self.assertIn('CVRP 최적 경로 계산 완료', response.context.get('success_message', ''))
        self.assertContains(response, "결과 요약")
        self.assertContains(response, "차량별 최적 경로 및 적재량")
        self.assertContains(response, "경로 시각화 (CVRP)")


class PdpDemoTests(TestCase):
    """PDP(픽업 및 배송 문제) 데모에 대한 테스트"""

    def test_introduction_view_loads_successfully(self):
        url = reverse('routing_logistics_app:pdp_introduction')
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'routing_logistics_app/pdp_introduction.html')
        self.assertContains(response, '수거 및 배송 문제 (Pickup and Delivery Problem - PDP)')

    def test_demo_view_loads_successfully(self):
        url = reverse('routing_logistics_app:pdp_demo')
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'routing_logistics_app/pdp_demo.html')
        self.assertContains(response, 'Pickup & Delivery Problem (PDP) Demo - Graphical Input')

    def test_demo_post_request_returns_solution(self):
        """PDP 데모가 POST 요청 시 픽업-배송 쌍을 고려한 최적 경로를 계산하는지 테스트합니다."""
        url = reverse('routing_logistics_app:pdp_demo')

        # PDP의 특성에 맞는 POST 데이터 (픽업-배송 쌍 정보)
        post_data = {
            'problem_type': 'pdp',
            'num_pairs': '3',
            'num_vehicles': '3',
            'vehicle_capacity': '100',
            'depot_x': '300.0',
            'depot_y': '250.0',
            'pair_0_id': 'Pair1',
            'pair_0_px': '103.0', 'pair_0_py': '120.0',
            'pair_0_dx': '510.0', 'pair_0_dy': '150.0',
            'pair_0_demand': '30',
            'pair_1_id': 'Pair2',
            'pair_1_px': '171.0', 'pair_1_py': '317.0',
            'pair_1_dx': '486.0', 'pair_1_dy': '283.0',
            'pair_1_demand': '40',
            'pair_2_id': 'Pair3',
            'pair_2_px': '384.0', 'pair_2_py': '45.0',
            'pair_2_dx': '302.0', 'pair_2_dy': '145.0',
            'pair_2_demand': '30',
        }

        response = self.client.post(url, post_data)
        self.assertEqual(response.status_code, 200)
        if settings.SAVE_DATA_FILE:
            self.assertIn("json'으로 서버에 저장되었습니다.", response.context.get('success_save_message', ''))
        # self.assertIsNotNone(response.context.get('opt_results'))
        self.assertIn('PDP 최적 경로 계산 완료', response.context.get('success_message', ''))
        self.assertContains(response, "결과 요약")
        self.assertContains(response, "차량별 최적 경로 및 적재량 변화")
        self.assertContains(response, "경로 시각화 (PDP)")