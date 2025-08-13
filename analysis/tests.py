from django.test import TestCase
from django.urls import reverse
from django.conf import settings
from analysis.models import Variable, Equation, MatrixEntry
from unittest import skipIf

# settings.SAVE_MODEL_DB가 False일 경우, 테스트를 건너뛰도록 설정
# 이렇게 하면 프로덕션 환경에서도 불필요한 DB 저장을 막을 수 있습니다.
# 예를 들어, Gurobi 라이센스 문제로 SAVE_MODEL_DB를 False로 설정할 수 있습니다.
@skipIf(not settings.SAVE_MODEL_DB, "SAVE_MODEL_DB is False in settings.")
class SportsSchedulingAnalysisTests(TestCase):
    def setUp(self):
        """테스트에 필요한 공통 데이터를 설정합니다."""
        self.url = reverse('puzzles_logic_app:sports_scheduling_demo')
        # 4개 팀 Gurobi 모델 데이터
        self.post_data_gurobi = {
            'problem_type': 'sports_scheduling',
            'num_teams': '4',
            'schedule_type': 'double',
            'max_consecutive': '3',
            'solver_type': 'GUROBI',
            'objective_choice': 'minimize_travel',
            'team_0_name': '한화', 'team_1_name': 'LG',
            'team_2_name': '롯데', 'team_3_name': 'KIA',
            'run_id': 'sports-min-dou-4Teams-GU-3con',
        }

        # 4개 팀 OR-Tools 모델 데이터
        self.post_data_ortools = self.post_data_gurobi.copy()
        self.post_data_ortools['solver_type'] = 'ORTOOLS'
        self.post_data_ortools['run_id'] = 'sports-min-dou-4Teams-OR-3con'

    def test_gurobi_model_data_saved_correctly(self):
        """Gurobi 솔버로 실행 시, 모델 데이터가 DB에 정확히 저장되는지 테스트합니다."""
        # 1. POST 요청 시뮬레이션
        response = self.client.post(self.url, self.post_data_gurobi)
        self.assertEqual(response.status_code, 200)

        # 2. run_id 추출 및 데이터 검증
        run_id = self.post_data_gurobi['run_id']

        # 3. DB에 저장된 데이터의 개수 검증
        # (4팀, 더블 라운드 로빈 기준)
        self.assertEqual(Variable.objects.filter(run_id=run_id).count(), 172)
        self.assertEqual(Equation.objects.filter(run_id=run_id).count(), 180)
        self.assertEqual(MatrixEntry.objects.filter(run_id=run_id).count(), 840)

        # 4. 특정 변수와 제약 조건의 존재 여부 및 값 검증
        play_var = Variable.objects.get(run_id=run_id, var_name='Plays_1_한화_LG')
        self.assertIsNotNone(play_var)
        self.assertEqual(play_var.var_group, 'Plays')

        travel_var = Variable.objects.get(run_id=run_id, var_name='Team_travel_한화')
        self.assertIsNotNone(travel_var)
        self.assertIsInstance(travel_var.result_value, float)
        self.assertEqual(travel_var.result_value, 0)

        # 5. 제약 조건 검증
        play_once_constr = Equation.objects.get(run_id=run_id, eq_name='PlayOnce_1_한화')
        self.assertEqual(play_once_constr.sign, '==')
        self.assertEqual(play_once_constr.rhs, 1.0)
        self.assertIsNotNone(play_once_constr)

    def test_ortools_model_data_saved_correctly(self):
        """OR-Tools 솔버로 실행 시, 모델 데이터가 DB에 정확히 저장되는지 테스트합니다."""
        response = self.client.post(self.url, self.post_data_ortools)
        self.assertEqual(response.status_code, 200)

        run_id = self.post_data_ortools['run_id']

        self.assertEqual(Variable.objects.filter(run_id=run_id).count(), 168)
        #self.assertGreater(Equation.objects.filter(run_id=run_id).count(), 0)
        #self.assertEqual(MatrixEntry.objects.filter(run_id=run_id).count(), 0)

        # 변수 및 제약 조건의 존재 여부 검증
        play_var = Variable.objects.get(run_id=run_id, var_name='plays_1_한화_LG')
        self.assertIsNotNone(play_var)
        self.assertEqual(play_var.var_group, 'Plays')

        Atloc_var_name = 'Atloc_한화_1_1'  # Ortools는 변수 이름이 다를 수 있음
        Atloc_var = Variable.objects.get(run_id=run_id, var_name=Atloc_var_name)
        self.assertEqual(Atloc_var.result_value, 1)