import logging
from ortools.sat.python import cp_model
from common_utils.ortools_solvers import BaseOrtoolsCpSolver
from core.decorators import log_solver_make

logger = logging.getLogger('production_scheduling_app')


class JobShopSolver(BaseOrtoolsCpSolver):
    """
    Job Shop Scheduling 문제를 정의하고 해결하는 클래스.
    """

    def __init__(self, input_data):
        """
        생성자: BaseCpSolver를 초기화하고 Job Shop 문제에 특화된 데이터를 파싱합니다.
        """
        super().__init__(input_data)

        # --- 1. 데이터 파싱 및 가공 ---
        self.jobs_data = self.input_data.get('jobs', [])
        self.num_jobs = len(self.jobs_data)
        self.num_machines = self.input_data.get('num_machines', 0)

        if self.num_jobs == 0 or self.num_machines == 0:
            raise ValueError("작업 또는 기계 데이터가 없습니다.")

        # 모델링에 사용할 변수 초기화
        self.all_tasks = {}  # (job_idx, op_idx) -> interval_var

    @log_solver_make
    def _create_variables(self):
        """작업별, 공정별 시작, 종료, 간격 변수를 생성합니다."""
        logger.solve("--- 1. Creating Job Shop Variables ---")

        # Horizon: 모든 작업 시간의 합
        horizon = sum(task[1] for job in self.jobs_data for task in job)

        for i, job in enumerate(self.jobs_data):
            for j, task in enumerate(job):
                machine_id, duration = task
                start_var = self.model.NewIntVar(0, horizon, f'start_{i}_{j}')
                end_var = self.model.NewIntVar(0, horizon, f'end_{i}_{j}')
                interval_var = self.model.NewIntervalVar(start_var, duration, end_var, f'interval_{i}_{j}')
                # (job_idx, op_idx)를 키로 사용하여 interval_var 저장
                self.all_tasks[(i, j)] = interval_var

    @log_solver_make
    def _add_constraints(self):
        """기계 독점 제약 및 작업 내 공정 순서 제약을 추가합니다."""
        logger.solve("--- 2. Adding Job Shop Constraints ---")

        # 제약 1: 기계 독점 제약 (No Overlap)
        # 각 기계별로 해당 기계를 사용하는 모든 작업들을 모아서 겹치지 않도록 설정
        for j in range(self.num_machines):
            intervals_on_machine = []
            for i, job in enumerate(self.jobs_data):
                for k, task in enumerate(job):
                    if task[0] == j:  # 이 공정이 현재 기계(j)를 사용한다면
                        intervals_on_machine.append(self.all_tasks[(i, k)])
            self.model.AddNoOverlap(intervals_on_machine)

        # 제약 2: 작업 내 공정 순서 제약 (Precedence)
        # 한 작업의 다음 공정은 이전 공정이 끝나야 시작 가능
        for i in range(self.num_jobs):
            for j in range(len(self.jobs_data[i]) - 1):
                self.model.Add(self.all_tasks[(i, j + 1)].StartExpr() >= self.all_tasks[(i, j)].EndExpr())

    @log_solver_make
    def _set_objective_function(self):
        """Makespan (최종 완료 시간) 최소화를 목표 함수로 설정합니다."""
        logger.solve("--- 3. Setting Objective Function (Minimize Makespan) ---")

        horizon = sum(task[1] for job in self.jobs_data for task in job)
        makespan = self.model.NewIntVar(0, horizon, 'makespan')

        # Makespan은 모든 작업의 '마지막 공정'이 끝나는 시간 중 가장 늦은 시간
        last_op_end_times = [
            self.all_tasks[(i, len(self.jobs_data[i]) - 1)].EndExpr()
            for i in range(self.num_jobs)
        ]
        self.model.AddMaxEquality(makespan, last_op_end_times)
        self.model.Minimize(makespan)

    def _extract_results(self, solver):
        """솔버 실행 후, 최적 스케줄과 Makespan을 추출합니다."""
        logger.info("Extracting results for Job Shop Scheduling...")

        job_schedules = []
        for i in range(self.num_jobs):
            job_schedule = {'job_id': self.input_data['job_ids'][i], 'tasks': []}

            # 작업 내 공정 순서대로 정렬
            sorted_ops = sorted(range(len(self.jobs_data[i])),
                                key=lambda op_idx: solver.Value(self.all_tasks[(i, op_idx)].StartExpr()))

            for j in sorted_ops:
                machine_id = self.jobs_data[i][j][0]
                start_time = solver.Value(self.all_tasks[(i, j)].StartExpr())
                end_time = solver.Value(self.all_tasks[(i, j)].EndExpr())
                job_schedule['tasks'].append({
                    'machine': f'Machine {machine_id + 1}',
                    'start': start_time,
                    'end': end_time,
                    'duration': end_time - start_time
                })
            job_schedules.append(job_schedule)

        return {
            'schedule': job_schedules,
            'makespan': solver.ObjectiveValue(),
        }
