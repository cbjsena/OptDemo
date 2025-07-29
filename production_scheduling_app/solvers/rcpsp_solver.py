import logging
from ortools.sat.python import cp_model
from common_utils.ortools_solvers import BaseOrtoolsCpSolver
from core.decorators import log_solver_make

logger = logging.getLogger('production_scheduling_app')


class RcpspSolver(BaseOrtoolsCpSolver):
    """
    RCPSP(Resource-Constrained Project Scheduling Problem)를 정의하고 해결하는 클래스.
    """

    def __init__(self, input_data):
        """
        생성자: BaseCpSolver를 초기화하고 RCPSP에 특화된 데이터를 파싱합니다.
        """
        super().__init__(input_data)

        # --- 1. 데이터 파싱 및 가공 ---
        self.activities_data = self.input_data.get('activities', [])
        self.resource_availabilities = self.input_data.get('resource_availabilities', [])
        self.num_activities = len(self.activities_data)
        self.num_resources = len(self.resource_availabilities)

        if self.num_activities == 0 or self.num_resources == 0:
            raise ValueError("활동 또는 자원 데이터가 없습니다.")

        # 모델링에 사용할 변수 초기화
        self.start_vars = []
        self.end_vars = []
        self.interval_vars = []

    @log_solver_make
    def _create_variables(self):
        """활동별 시작, 종료, 간격 변수를 생성합니다."""
        # Horizon: 모든 활동 기간의 합
        horizon = sum(act['duration'] for act in self.activities_data)

        self.start_vars = [self.model.NewIntVar(0, horizon, f'start_{i}') for i in range(self.num_activities)]
        self.end_vars = [self.model.NewIntVar(0, horizon, f'end_{i}') for i in range(self.num_activities)]
        self.interval_vars = [
            self.model.NewIntervalVar(self.start_vars[i],self.activities_data[i]['duration'],
                                      self.end_vars[i],f'interval_{i}')
            for i in range(self.num_activities)
        ]

    @log_solver_make
    def _add_constraints(self):
        """선후 관계 제약 및 자원 제약을 추가합니다."""
        # 제약 1: 선후 관계 제약 (Precedence)
        for i in range(self.num_activities):
            for pred_idx in self.activities_data[i]['predecessors']:
                # 선행 작업의 인덱스는 1-based이므로 0-based로 변환
                self.model.Add(self.start_vars[i] >= self.end_vars[pred_idx])

        # 제약 2: 자원 제약 (Cumulative)
        for k in range(self.num_resources):
            demands = [act['resource_reqs'][k] for act in self.activities_data]
            self.model.AddCumulative(self.interval_vars, demands, self.resource_availabilities[k])

    @log_solver_make
    def _set_objective_function(self):
        """Makespan (프로젝트 최종 완료 시간) 최소화를 목표 함수로 설정합니다."""
        horizon = sum(act['duration'] for act in self.activities_data)
        makespan = self.model.NewIntVar(0, horizon, 'makespan')

        self.model.AddMaxEquality(makespan, self.end_vars)
        self.model.Minimize(makespan)

    def _extract_results(self, solver):
        """솔버 실행 후, 최적 스케줄, Makespan, 자원 사용량 프로필을 추출합니다."""
        logger.info("Extracting results for RCPSP...")

        makespan_value = int(solver.ObjectiveValue())
        schedule = []
        for i in range(self.num_activities):
            schedule.append({
                'id': self.activities_data[i]['id'],
                'start': solver.Value(self.start_vars[i]),
                'end': solver.Value(self.end_vars[i]),
                'duration': self.activities_data[i]['duration'],
                'resource_reqs': self.activities_data[i]['resource_reqs']
            })

        schedule.sort(key=lambda item: item['start'])

        # 자원 사용량 프로필 계산
        resource_usage = {}
        for k in range(self.num_resources):
            usage_profile = [0] * (makespan_value + 1)
            for t in range(makespan_value):
                for i in range(self.num_activities):
                    if solver.Value(self.start_vars[i]) <= t < solver.Value(self.end_vars[i]):
                        usage_profile[t] += self.activities_data[i]['resource_reqs'][k]
            resource_usage[f'Resource {k + 1}'] = usage_profile

        return {
            'schedule': schedule,
            'makespan': makespan_value,
            'resource_usage': resource_usage,
        }
