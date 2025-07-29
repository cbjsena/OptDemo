import logging
from common_utils.ortools_solvers import BaseOrtoolsCpSolver
from core.decorators import log_solver_make

logger = logging.getLogger('production_scheduling_app')

# --- 고정된 순서에 대한 Makespan 계산 함수 (새로 추가) ---
def calculate_flow_shop_schedule(processing_times, job_ids, sequence):
    """
    주어진 작업 순서(sequence)에 따라 Flow Shop 스케줄과 Makespan을 계산합니다.
    processing_times: [[p_ij, ...], ...]
    job_ids: ['Job 1', 'Job 2', ...]
    sequence: 순서를 나타내는 job_id 리스트. 예: ['Job 2', 'Job 1', 'Job 3']
    """
    num_jobs = len(processing_times)
    num_machines = len(processing_times[0]) if num_jobs > 0 else 0

    # job_id를 인덱스로 변환
    job_id_to_index = {job_id: i for i, job_id in enumerate(job_ids)}
    try:
        sequence_indices = [job_id_to_index[job_id] for job_id in sequence]
    except KeyError as e:
        raise ValueError(f"잘못된 작업 ID가 수동 순서에 포함되어 있습니다: {e}")

    if len(sequence_indices) != num_jobs or len(set(sequence_indices)) != num_jobs:
        raise ValueError("수동 순서에는 모든 작업이 정확히 한 번씩 포함되어야 합니다.")

    # 완료 시간 행렬 C_ij 초기화
    completion_times = [[0] * num_machines for _ in range(num_jobs)]

    # 재귀적 관계를 사용하여 완료 시간 계산
    for k in range(num_jobs):  # 순서 k (0 to n-1)
        job_idx = sequence_indices[k]
        for j in range(num_machines):  # 기계 j (0 to m-1)
            # 첫 번째 작업(k=0) 또는 첫 번째 기계(j=0)의 완료 시간
            prev_job_completion_on_same_machine = completion_times[sequence_indices[k - 1]][j] if k > 0 else 0
            prev_machine_completion_for_same_job = completion_times[job_idx][j - 1] if j > 0 else 0

            completion_times[job_idx][j] = max(prev_job_completion_on_same_machine,
                                               prev_machine_completion_for_same_job) + processing_times[job_idx][j]

    # Makespan은 마지막 순서의 작업이 마지막 기계에서 끝나는 시간
    makespan = completion_times[sequence_indices[-1]][num_machines - 1]

    # 간트 차트용 데이터 생성
    schedule = []
    for i in range(num_jobs):
        job_schedule = {'job_id': job_ids[i], 'tasks': []}
        for j in range(num_machines):
            end_time = completion_times[i][j]
            start_time = end_time - processing_times[i][j]
            job_schedule['tasks'].append({
                'machine': f'Machine {j + 1}',
                'start': start_time,
                'duration': processing_times[i][j],
                'end': end_time
            })
        schedule.append(job_schedule)
    results = {'schedule': schedule, 'makespan': makespan, 'sequence': sequence}

    return results


class FlowShopSolver(BaseOrtoolsCpSolver):
    """
    Flow Shop Scheduling 문제를 정의하고 해결하는 클래스.
    """

    def __init__(self, input_data):
        """
        생성자: BaseCpSolver를 초기화하고 Flow Shop 문제에 특화된 데이터를 파싱합니다.
        """
        super().__init__(input_data)

        # --- 1. 데이터 파싱 및 가공 ---
        self.processing_times = self.input_data.get('processing_times', [])
        self.job_ids = self.input_data.get('job_ids', [])
        self.num_jobs = len(self.processing_times)
        self.num_machines = len(self.processing_times[0]) if self.num_jobs > 0 else 0

        if self.num_jobs == 0 or self.num_machines == 0:
            raise ValueError("작업 또는 기계 데이터가 없습니다.")

        # 모델링에 사용할 변수 초기화
        self.completion_times = []
        self.start_vars = []
        self.interval_vars = []

    @log_solver_make
    def _create_variables(self):
        """작업별, 기계별 완료 시간 및 간격 변수를 생성합니다."""
        logger.solve("--- 1. Creating Flow Shop Variables ---")

        # Horizon: 모든 작업 시간의 합
        horizon = sum(sum(job) for job in self.processing_times)

        self.completion_times = [
            [self.model.NewIntVar(0, horizon, f'C_{i}_{j}') for j in range(self.num_machines)]
            for i in range(self.num_jobs)
        ]

        # NoOverlap 제약을 위한 start, interval 변수 추가 생성
        self.start_vars = [
            [self.model.NewIntVar(0, horizon, f'start_{i}_{j}') for j in range(self.num_machines)]
            for i in range(self.num_jobs)
        ]

        self.interval_vars = [
            [self.model.NewIntervalVar(
                self.start_vars[i][j],
                self.processing_times[i][j],
                self.completion_times[i][j],
                f'interval_{i}_{j}'
            ) for j in range(self.num_machines)]
            for i in range(self.num_jobs)
        ]

    @log_solver_make
    def _add_constraints(self):
        """기계 순서(작업 흐름) 및 작업 순서(기계 독점) 제약을 추가합니다."""
        logger.solve("--- 2. Adding Flow Shop Constraints ---")

        # 제약 1: 기계 순서 제약 (작업 흐름)
        # 한 작업의 다음 공정은 이전 공정이 끝나야 시작 가능
        for i in range(self.num_jobs):
            for j in range(1, self.num_machines):
                self.model.Add(self.start_vars[i][j] >= self.completion_times[i][j - 1])

        # 제약 2: 작업 순서 제약 (기계 독점)
        # 각 기계에서는 작업들이 겹칠 수 없음
        for j in range(self.num_machines):
            intervals_on_machine = [self.interval_vars[i][j] for i in range(self.num_jobs)]
            self.model.AddNoOverlap(intervals_on_machine)

    @log_solver_make
    def _set_objective_function(self):
        """Makespan (최종 완료 시간) 최소화를 목표 함수로 설정합니다."""
        logger.solve("--- 3. Setting Objective Function (Minimize Makespan) ---")

        horizon = sum(sum(job) for job in self.processing_times)
        makespan = self.model.NewIntVar(0, horizon, 'makespan')

        # Makespan은 모든 작업이 마지막 기계에서 끝나는 시간 중 가장 늦은 시간
        last_machine_completions = [self.completion_times[i][self.num_machines - 1] for i in range(self.num_jobs)]
        self.model.AddMaxEquality(makespan, last_machine_completions)
        self.model.Minimize(makespan)

    def _extract_results(self, solver):
        """솔버 실행 후, 최적 순서와 스케줄, Makespan을 추출합니다."""
        logger.info("Extracting results for Flow Shop Scheduling...")

        # 첫 번째 기계의 시작 시간을 기준으로 최적 순서 결정
        sequence_info = []
        for i in range(self.num_jobs):
            start_time_on_m0 = solver.Value(self.start_vars[i][0])
            sequence_info.append({'job_index': i, 'start_time': start_time_on_m0})

        sequence_info.sort(key=lambda item: item['start_time'])

        optimal_sequence_indices = [item['job_index'] for item in sequence_info]
        optimal_sequence_ids = [self.job_ids[i] for i in optimal_sequence_indices]

        # CP-SAT 결과는 휴리스틱 기반일 수 있으므로, 결정된 순서로 정확한 스케줄을 재계산하여 반환
        # (calculate_flow_shop_schedule는 기존 함수를 재활용)
        results = calculate_flow_shop_schedule(
            self.processing_times,
            self.job_ids,
            optimal_sequence_ids
        )
        return results
