from django.test import TestCase
from django.urls import reverse


class ResourceAllocationAppTests(TestCase):

    def test_production_scheduling_introduction_view_loads_successfully(self):
        """Production Scheduling 데모 케이스 설명 페이지가 정상적으로 로드되는지 테스트합니다."""
        url = reverse('production_scheduling_app:production_scheduling_introduction')
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'production_scheduling_app/production_scheduling_introduction.html')
        self.assertContains(response, 'Lot Sizing Problem')
        self.assertContains(response, 'Single Machine Scheduling')
        self.assertContains(response, 'Flow Shop Scheduling')
        self.assertContains(response, 'Job Shop Scheduling')
        self.assertContains(response, 'Project Scheduling (RCPSP)')


    def test_lot_sizing_introduction_view_loads_successfully(self):
        """Lot Sizing 설명 페이지가 GET 요청 시 정상적으로 로드되는지 테스트합니다."""
        url = reverse('production_scheduling_app:lot_sizing_introduction')
        response = self.client.get(url)

        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'production_scheduling_app/lot_sizing_introduction.html')
        self.assertContains(response, '생산 로트 크기 결정 문제')


    def test_lot_sizing_demo_view_loads_successfully(self):
        """Lot Sizing 데모 페이지가 GET 요청 시 정상적으로 로드되는지 테스트합니다."""
        url = reverse('production_scheduling_app:lot_sizing_demo')
        response = self.client.get(url)

        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'production_scheduling_app/lot_sizing_demo.html')
        self.assertContains(response, 'Lot Sizing Problem Demo')

    def test_lot_sizing_demo_post_request_returns_solution(self):
        """Lot Sizing 데모가 POST 요청 시 최적 생산 계획을 계산하는지 테스트합니다."""
        url = reverse('production_scheduling_app:lot_sizing_demo')

        # 제공된 로그를 기반으로 POST 요청 데이터 생성
        post_data = {
            'problem_type': 'lot_sizing',
            'num_periods': '6',
            'demand_0': '145', 'demand_1': '60', 'demand_2': '116',
            'demand_3': '146', 'demand_4': '69', 'demand_5': '125',
            'setup_cost_0': '321', 'setup_cost_1': '319', 'setup_cost_2': '230',
            'setup_cost_3': '224', 'setup_cost_4': '430', 'setup_cost_5': '435',
            'prod_cost_0': '9', 'prod_cost_1': '9', 'prod_cost_2': '5',
            'prod_cost_3': '12', 'prod_cost_4': '7', 'prod_cost_5': '8',
            'holding_cost_0': '2', 'holding_cost_1': '5', 'holding_cost_2': '3',
            'holding_cost_3': '4', 'holding_cost_4': '5', 'holding_cost_5': '5',
            'capacity_0': '243', 'capacity_1': '156', 'capacity_2': '218',
            'capacity_3': '265', 'capacity_4': '181', 'capacity_5': '290'
        }

        # POST 요청 시뮬레이션
        response = self.client.post(url, post_data)

        # 결과 검증
        self.assertEqual(response.status_code, 200)
        self.assertIsNotNone(response.context.get('results'))
        self.assertIn('최적 생산 계획 수립 완료', response.context.get('success_message', ''))
        self.assertContains(response, "최소 총 비용")
        self.assertContains(response, "최적 생산 및 재고 계획")

    def test_single_machine_introduction_view_loads_successfully(self):
        """Single Machine Scheduling 설명 페이지가 GET 요청 시 정상적으로 로드되는지 테스트합니다."""
        url = reverse('production_scheduling_app:single_machine_introduction')
        response = self.client.get(url)

        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'production_scheduling_app/single_machine_introduction.html')
        self.assertContains(response, '단일 기계 스케줄링')


    def test_single_machine_demo_view_loads_successfully(self):
        """Single Machine Scheduling 데모 페이지가 GET 요청 시 정상적으로 로드되는지 테스트합니다."""
        url = reverse('production_scheduling_app:single_machine_demo')
        response = self.client.get(url)

        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'production_scheduling_app/single_machine_demo.html')
        self.assertContains(response, 'Single Machine Scheduling Demo')

    def test_single_machine_demo_post_request_returns_solution(self):
        """Single Machine Scheduling 데모가 POST 요청 시 최적 스케줄을 계산하는지 테스트합니다."""
        url = reverse('production_scheduling_app:single_machine_demo')

        post_data = {
            'problem_type': 'single_machine',
            'num_jobs': '5',
            'objective_choice': 'total_flow_time',
            'job_0_id': 'Job 1',
            'job_0_processing_time': '7',
            'job_0_due_date': '20',
            'job_0_release_time': '0',
            'job_1_id': 'Job 2',
            'job_1_processing_time': '3',
            'job_1_due_date': '5',
            'job_1_release_time': '0',
            'job_2_id': 'Job 3',
            'job_2_processing_time': '5',
            'job_2_due_date': '25',
            'job_2_release_time': '6',
            'job_3_id': 'Job 4',
            'job_3_processing_time': '4',
            'job_3_due_date': '10',
            'job_3_release_time': '0',
            'job_4_id': 'Job 5',
            'job_4_processing_time': '1',
            'job_4_due_date': '30',
            'job_4_release_time': '0',
        }

        # POST 요청 시뮬레이션
        response = self.client.post(url, post_data)

        # 결과 검증
        self.assertEqual(response.status_code, 200)
        self.assertIsNotNone(response.context.get('results'))
        self.assertIn('최적 스케줄 계산 완료', response.context.get('success_message', ''))
        self.assertContains(response, "결과 요약")
        self.assertContains(response, "간트 차트 (Gantt Chart)")

    def test_single_machine_advanced_view_loads_successfully(self):
        """Single Machine Scheduling Advanced 페이지가 GET 요청 시 정상적으로 로드되는지 테스트합니다."""
        url = reverse('production_scheduling_app:single_machine_advanced')
        response = self.client.get(url)

        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'production_scheduling_app/single_machine_advanced.html')
        self.assertContains(response, '단일 기계 스케줄링 문제 (Single Machine Scheduling Problem) - Advanced')


    def test_flow_shop_introduction_view_loads_successfully(self):
        """Flow Shop Scheduling 설명 페이지가 GET 요청 시 정상적으로 로드되는지 테스트합니다."""
        url = reverse('production_scheduling_app:flow_shop_introduction')
        response = self.client.get(url)

        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'production_scheduling_app/flow_shop_introduction.html')
        self.assertContains(response, '흐름 공정 스케줄링 (Flow Shop Scheduling)')


    def test_flow_shop_demo_view_loads_successfully(self):
        """Flow Shop Scheduling 데모 페이지가 GET 요청 시 정상적으로 로드되는지 테스트합니다."""
        url = reverse('production_scheduling_app:flow_shop_demo')
        response = self.client.get(url)

        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'production_scheduling_app/flow_shop_demo.html')
        self.assertContains(response, 'Flow Shop Scheduling Demo')

    def test_flow_shop_demo_post_request_returns_solution(self):
        """Flow shop 데모가 POST 요청 시 최적 생산 계획을 계산하는지 테스트합니다."""
        url = reverse('production_scheduling_app:flow_shop_demo')

        post_data = {
            'problem_type': 'flow_shop',
            'action': 'optimize',
            'num_jobs': '4',
            'num_machines': '3',
            'job_0_id': 'Job_1',
            'p_0_0': '29',
            'p_0_1': '78',
            'p_0_2': '9',
            'job_1_id': 'Job_2',
            'p_1_0': '43',
            'p_1_1': '92',
            'p_1_2': '8',
            'job_2_id': 'Job_3',
            'p_2_0': '90',
            'p_2_1': '85',
            'p_2_2': '87',
            'job_3_id': 'Job_4',
            'p_3_0': '77',
            'p_3_1': '39',
            'p_3_2': '55',
        }

        # POST 요청 시뮬레이션
        response = self.client.post(url, post_data)

        # 결과 검증
        self.assertEqual(response.status_code, 200)
        self.assertIsNotNone(response.context.get('results'))
        self.assertIn('최적 스케줄 계산 완료', response.context.get('success_message', ''))
        self.assertContains(response, "결과 요약")
        self.assertContains(response, "[최적화 결과] 간트 차트 (Gantt Chart)")
        # 결과에 makespan 값이 포함되어 있는지 확인
        results = response.context.get('results', {})
        self.assertIn('makespan', results)

    def test_job_shop_introduction_view_loads_successfully(self):
        """Job Shop Scheduling 설명 페이지가 GET 요청 시 정상적으로 로드되는지 테스트합니다."""
        url = reverse('production_scheduling_app:job_shop_introduction')
        response = self.client.get(url)

        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'production_scheduling_app/job_shop_introduction.html')
        self.assertContains(response, '작업장 일정 계획 문제 (Job Shop Scheduling)')

    def test_job_shop_demo_view_loads_successfully(self):
        """Job Shop Scheduling 데모 페이지가 GET 요청 시 정상적으로 로드되는지 테스트합니다."""
        url = reverse('production_scheduling_app:job_shop_demo')
        response = self.client.get(url)

        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'production_scheduling_app/job_shop_demo.html')
        self.assertContains(response, 'Job Shop Scheduling Demo')

    def test_job_shop_demo_post_request_returns_solution(self):
        """Job shop 데모가 POST 요청 시 최적 생산 계획을 계산하는지 테스트합니다."""
        url = reverse('production_scheduling_app:job_shop_demo')

        post_data = {
            'problem_type': 'job_shop',
            'num_jobs': '4',
            'num_machines': '3',
            'job_0_id': 'Job_1',
            'p_0_0': '29', 'p_0_1': '78', 'p_0_2': '9',
            'job_0_routing': '0-1-2',
            'job_1_id': 'Job_2',
            'p_1_0': '43', 'p_1_1': '92', 'p_1_2': '8',
            'job_1_routing': '2-0-1',
            'job_2_id': 'Job_3',
            'p_2_0': '90', 'p_2_1': '85', 'p_2_2': '87',
            'job_2_routing': '1-2-0',
            'job_3_id': 'Job_4',
            'p_3_0': '77', 'p_3_1': '39', 'p_3_2': '55',
            'job_3_routing': '2-0-1',
        }

        # POST 요청 시뮬레이션
        response = self.client.post(url, post_data)

        # 결과 검증
        self.assertEqual(response.status_code, 200)
        self.assertIsNotNone(response.context.get('results'))
        self.assertIn('최적 스케줄 계산 완료', response.context.get('success_message', ''))
        self.assertContains(response, "결과 요약")
        self.assertContains(response, "간트 차트 (Gantt Chart)")
        self.assertContains(response, "Makespan")

    def test_rcpsp_introduction_view_loads_successfully(self):
        """RCPSP 설명 페이지가 GET 요청 시 정상적으로 로드되는지 테스트합니다."""
        url = reverse('production_scheduling_app:rcpsp_introduction')
        response = self.client.get(url)

        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'production_scheduling_app/rcpsp_introduction.html')
        self.assertContains(response, '프로젝트 일정 및 자원 제약 스케줄링 (RCPSP)')


    def test_rcpsp_demo_view_loads_successfully(self):
        """RCPSP 데모 페이지가 GET 요청 시 정상적으로 로드되는지 테스트합니다."""
        url = reverse('production_scheduling_app:rcpsp_demo')
        response = self.client.get(url)

        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'production_scheduling_app/rcpsp_demo.html')
        self.assertContains(response, 'Resource-Constrained Project Scheduling (RCPSP) Demo')

    def test_rcpsp_demo_post_request_returns_solution(self):
        """RCPSP 데모가 POST 요청 시 최적 생산 계획을 계산하는지 테스트합니다."""
        url = reverse('production_scheduling_app:rcpsp_demo')

        post_data = {
            'problem_type': 'rcpsp',
            'num_activities': '8',
            'num_resources': '3',
            'resource_0_name': 'Front-End 개발자', 'resource_0_availability': '4',
            'resource_1_name': 'Back-End 개발자', 'resource_1_availability': '5',
            'resource_2_name': 'QA 엔지니어', 'resource_2_availability': '3',
            'activity_0_id': 'A.기획/설계', 'activity_0_duration': '5', 'activity_0_predecessors': '',
            'activity_0_res_0_req': '1', 'activity_0_res_1_req': '1', 'activity_0_res_2_req': '1',
            'activity_1_id': 'B.UI/UX디자인', 'activity_1_duration': '4', 'activity_1_predecessors': '1',
            'activity_1_res_0_req': '2', 'activity_1_res_1_req': '0', 'activity_1_res_2_req': '1',
            'activity_2_id': 'C.DB설계', 'activity_2_duration': '3', 'activity_2_predecessors': '1',
            'activity_2_res_0_req': '0', 'activity_2_res_1_req': '2', 'activity_2_res_2_req': '0',
            'activity_3_id': 'D.FE개발', 'activity_3_duration': '8', 'activity_3_predecessors': '2',
            'activity_3_res_0_req': '3', 'activity_3_res_1_req': '0', 'activity_3_res_2_req': '0',
            'activity_4_id': 'E.BE개발', 'activity_4_duration': '7', 'activity_4_predecessors': '3',
            'activity_4_res_0_req': '0', 'activity_4_res_1_req': '4', 'activity_4_res_2_req': '0',
            'activity_5_id': 'F.통합테스트', 'activity_5_duration': '4', 'activity_5_predecessors': '4,5',
            'activity_5_res_0_req': '1', 'activity_5_res_1_req': '1', 'activity_5_res_2_req': '3',
            'activity_6_id': 'G.피드백반영', 'activity_6_duration': '3', 'activity_6_predecessors': '6',
            'activity_6_res_0_req': '2', 'activity_6_res_1_req': '2', 'activity_6_res_2_req': '1',
            'activity_7_id': 'H.최종배포', 'activity_7_duration': '2', 'activity_7_predecessors': '7',
            'activity_7_res_0_req': '1', 'activity_7_res_1_req': '1', 'activity_7_res_2_req': '2',
        }

        # POST 요청 시뮬레이션
        response = self.client.post(url, post_data)

        # 결과 검증
        self.assertEqual(response.status_code, 200)
        self.assertIsNotNone(response.context.get('results'))
        self.assertIn('최적 프로젝트 스케줄 수립 완료', response.context.get('success_message', ''))
        self.assertContains(response, "결과 요약")
        self.assertContains(response, "최소 프로젝트 완료 기간 (Makespan)")
        self.assertContains(response, "간트 차트 (Gantt Chart)")
        self.assertContains(response, "자원 사용량 프로필 (Resource Profile)")
