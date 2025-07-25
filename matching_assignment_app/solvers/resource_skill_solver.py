import logging
from ortools.linear_solver import pywraplp
from common_utils.ortools_solvers import BaseOrtoolsLinearSolver

# from common_utils.common_run_opt import solving_log # solving_log가 있다고 가정

logger = logging.getLogger('matching_assignment_app')


class ResourceSkillMatchingSolver(BaseOrtoolsLinearSolver):
    """
    자원-기술 매칭 문제를 해결하여 총비용을 최소화하는 클래스.
    2단계 해결 방식을 사용하여 실행 불가능한 경우 원인을 먼저 파악합니다.
    """

    def __init__(self, input_data):
        """
        생성자: BaseOrtoolsLinearSolver를 초기화하고 문제 데이터를 파싱합니다.
        """
        super().__init__(input_data, 'CBC')  # MIP를 지원하는 CBC 솔버 사용

        self.resources_data = self.input_data.get('resources_data', [])
        self.projects_data = self.input_data.get('projects_data', [])
        self.num_resources = len(self.resources_data)
        self.num_projects = len(self.projects_data)

        # 결정 변수 초기화
        self.x = {}  # 할당 변수
        self.unfulfilled_skills = {}  # 실행 가능성 체크를 위한 슬랙 변수

    def _create_variables(self):
        """결정 변수 (할당 변수, 슬랙 변수)를 생성합니다."""
        logger.solve("--- 1a. Creating Assignment Variables ---")
        self.x = {(i, j): self.solver.BoolVar(f'x_{i}_{j}')
                  for i in range(self.num_resources)
                  for j in range(self.num_projects)}

        logger.solve("--- 1b. Creating Feasibility Slack Variables ---")
        for j in range(self.num_projects):
            for skill in self.projects_data[j].get('required_skills', []):
                self.unfulfilled_skills[j, skill] = self.solver.BoolVar(f'unfulfilled_{j}_{skill}')

    def _add_constraints(self):
        """모델에 모든 제약 조건들을 추가합니다."""
        logger.solve("--- 2. Adding Constraints ---")

        # 제약 1: 각 인력은 최대 하나의 프로젝트에만 할당
        for i in range(self.num_resources):
            self.solver.Add(sum(self.x[i, j] for j in range(self.num_projects)) <= 1)

        # 제약 2: 각 프로젝트의 기술 요구사항 충족 (슬랙 변수 포함)
        for j in range(self.num_projects):
            for skill in self.projects_data[j].get('required_skills', []):
                self.solver.Add(
                    sum(self.x[i, j] for i in range(self.num_resources) if
                        skill in self.resources_data[i].get('skills', []))
                    + self.unfulfilled_skills[j, skill] >= 1
                )

    def _set_objective_function(self):
        """
        [2단계용] 비용 최소화 목표 함수를 설정합니다.
        (1단계 목표는 solve 메서드에서 직접 처리)
        """
        logger.solve("--- 3. Setting Cost Minimization Objective ---")
        objective = self.solver.Objective()
        for i in range(self.num_resources):
            for j in range(self.num_projects):
                objective.SetCoefficient(self.x[i, j], self.resources_data[i].get('cost', 0))
        objective.SetMinimization()

    def _extract_results(self):
        """솔버 실행 후, 결과를 가공하여 딕셔너리 형태로 반환합니다."""
        logger.info("Extracting results for resource-skill matching...")

        assignments = {}
        assigned_resource_indices = set()
        for j in range(self.num_projects):
            project_name = self.projects_data[j].get('name')
            assignments[project_name] = []
            for i in range(self.num_resources):
                if self.x[i, j].solution_value() > 0.5:
                    assignments[project_name].append(self.resources_data[i])
                    assigned_resource_indices.add(i)

        unassigned_resources = [self.resources_data[i] for i in range(self.num_resources) if
                                i not in assigned_resource_indices]

        return {
            'assignments': assignments,
            'total_cost': self.solver.Objective().Value(),
            'unassigned_resources': unassigned_resources
        }

    def solve(self):
        """
        [재정의] 2단계 최적화 프로세스를 실행하는 메인 메서드.
        1단계: 실행 가능성 확인 (Unmet Skills 최소화)
        2단계: 비용 최소화
        """
        try:
            if self.num_resources == 0 or self.num_projects == 0:
                return None, "오류: 인력 또는 프로젝트 데이터가 없습니다.", 0.0

            self._create_variables()
            self._add_constraints()

            # --- 1단계: 실행 가능성 확인 ---
            logger.info("Phase 1: Solving for feasibility...")
            feasibility_objective = self.solver.Objective()
            for slack_var in self.unfulfilled_skills.values():
                feasibility_objective.SetCoefficient(slack_var, 1)
            feasibility_objective.SetMinimization()

            status = self.solver.Solve()

            # 실행 가능성 모델 해결 실패 시
            if status not in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
                return None, "오류: 실행 가능성 확인 모델을 푸는 데 실패했습니다.", 0.0

            # 충족되지 않은 기술 요구사항이 있는지 확인
            if feasibility_objective.Value() > 0:
                unmet_reqs = [f"'{self.projects_data[j]['name']}' 프로젝트의 '{skill}' 기술"
                              for (j, skill), var in self.unfulfilled_skills.items() if var.solution_value() > 0.5]
                error_msg = f"실행 불가능한 문제입니다. 다음 요구사항을 충족할 수 없습니다: {', '.join(unmet_reqs)}"
                logger.warning(f"Model is INFEASIBLE. Unmet skills: {unmet_reqs}")
                return None, error_msg, 0.0

            # --- 2단계: 비용 최소화 ---
            logger.info("Phase 2: Model is feasible. Solving for minimum cost...")
            # 슬랙 변수들을 0으로 고정
            for slack_var in self.unfulfilled_skills.values():
                self.solver.Add(slack_var == 0)

            # 비용 최소화 목표 설정
            self._set_objective_function()

            # 다시 해결
            status = self.solver.Solve()
            processing_time = self.get_time(self.solver.WallTime() / 1000.0)
            self.log_solve_resulte(self.status_map.get(status, "UNKNOWN"), processing_time)

            if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
                results = self._extract_results()
                error_msg = "실행 가능하지만 최적해는 아닐 수 있습니다." if status == pywraplp.Solver.FEASIBLE else None
                return results, error_msg, processing_time
            else:
                error_msg = f"최적 할당을 찾지 못했습니다. (솔버 상태: {status})"
                return None, error_msg, processing_time

        except Exception as e:
            logger.error(f"An unexpected error occurred in {self.__class__.__name__}: {e}", exc_info=True)
            return None, f"오류 발생: {str(e)}", 0.0
