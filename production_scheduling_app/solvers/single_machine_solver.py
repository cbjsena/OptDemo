import logging
from ortools.sat.python import cp_model
from common_utils.ortools_solvers import BaseOrtoolsCpSolver
from core.decorators import log_solver_make

logger = logging.getLogger('production_scheduling_app')


class SingleMachineSolver(BaseOrtoolsCpSolver):
    """
    단일 기계 스케줄링 문제를 정의하고 해결하는 클래스.
    """

    def __init__(self, input_data):
        """
        생성자: BaseCpSolver를 초기화하고 문제에 특화된 데이터를 파싱합니다.
        """
        super().__init__(input_data)

        # --- 1. 데이터 파싱 및 가공 ---
        self.jobs_list = self.input_data.get('jobs_list', [])
        self.num_jobs = len(self.jobs_list)
        if self.num_jobs == 0:
            raise ValueError("작업 데이터가 없습니다.")

        self.objective_choice = self.input_data.get('objective_choice')

        # 모델링에 사용할 변수 초기화
        self.start_vars = {}
        self.end_vars = {}
        self.interval_vars = {}
        self.all_tasks = []

    @log_solver_make
    def _create_variables(self):
        """작업별 시작, 종료, 간격 변수를 생성합니다."""

        # 모든 작업 시간의 합을 계산하여 변수 상한으로 사용
        all_processing_times = [j['processing_time'] for j in self.jobs_list]
        horizon = sum(all_processing_times) + sum(j['release_time'] for j in self.jobs_list)

        for i, job in enumerate(self.jobs_list):
            job_id = job.get('id', i)
            duration = job['processing_time']
            suffix = f'_{job_id}'
            self.start_vars[i] = self.model.NewIntVar(job['release_time'], horizon, f'start{suffix}')
            self.end_vars[i] = self.model.NewIntVar(0, horizon, f'end{suffix}')
            self.interval_vars[i] = self.model.NewIntervalVar(
                self.start_vars[i], duration, self.end_vars[i], f'interval{suffix}')
            self.all_tasks.append((job, self.start_vars[i], self.end_vars[i]))

    @log_solver_make
    def _add_constraints(self):
        """작업이 겹치지 않도록 하고, 작업 시작 가능 시간을 설정하는 제약을 추가합니다."""
        # 제약 1: 단일 기계에서는 모든 작업이 겹칠 수 없음
        self.model.AddNoOverlap(list(self.interval_vars.values()))

        # 제약 2: 각 작업은 정해진 시작 가능 시간 이후에만 시작 가능
        for i, job in enumerate(self.jobs_list):
            release_time = job.get('release_time', 0)
            if release_time > 0:
                self.model.Add(self.start_vars[i] >= release_time)

    @log_solver_make
    def _set_objective_function(self):
        """사용자가 선택한 목표 함수(Makespan, 총 흐름 시간, 총 지연 시간)를 설정합니다."""
        logger.solve(f"---Setting Objective Function: {self.objective_choice} ---")

        if self.objective_choice == 'makespan':
            # 목표 1: Makespan (최종 완료 시간) 최소화
            makespan = self.model.NewIntVar(0, max(job['due_date'] for job in self.jobs_list), 'makespan')
            self.model.AddMaxEquality(makespan, list(self.end_vars.values()))
            self.model.Minimize(makespan)

        elif self.objective_choice == 'total_tardiness':
            # 목표 2: Total Tardiness (총 지연 시간) 최소화
            tardiness_vars = []
            for i, job in enumerate(self.jobs_list):
                due_date = job.get('due_date', 0)
                tardiness = self.model.NewIntVar(0, 1000, f"tardiness_{i}")
                # T_i >= C_i - d_i (여기서 C_i는 i번째 작업의 end_var)
                self.model.Add(self.end_vars[i] - due_date <= tardiness)
                tardiness_vars.append(tardiness)
            self.model.Minimize(sum(tardiness_vars))

        else:  # 기본값: total_flow_time
            # 목표 3: Total Flow Time (총 흐름 시간) 최소화
            self.model.Minimize(sum(self.end_vars.values()))

    def _extract_results(self, solver):
        """솔버 실행 후, 최적 스케줄과 성과 지표를 추출합니다."""
        logger.info("Extracting results for Single Machine Scheduling...")

        schedule = []
        for i, job_data in enumerate(self.all_tasks):
            job, start_var, end_var = job_data
            schedule.append({
                'id': job['id'],
                'start': solver.Value(start_var),
                'end': solver.Value(end_var),
                'processing_time': job['processing_time'],
                'due_date': job['due_date'],
                'release_time': job['release_time']
            })

        # 시작 시간 기준으로 스케줄 정렬
        schedule.sort(key=lambda item: item['start'])

        # 성과 지표 계산
        makespan = max(job['end'] for job in schedule) if schedule else 0
        total_tardiness = sum(max(0, job['end'] - job['due_date']) for job in schedule)
        total_flow_time = sum(job['end'] - job['release_time'] for job in schedule)

        return {
            'schedule': schedule,
            'sequence': [job['id'] for job in schedule],
            'makespan': makespan,
            'total_flow_time': total_flow_time,
            'total_tardiness': total_tardiness,
            'objective_value': solver.ObjectiveValue()
        }
