import logging
import time

from django.conf import settings
from ortools.linear_solver import pywraplp
from ortools.sat.python import cp_model
from .base_solver import BaseSolver
from .common_run_opt import export_ortools_solver, export_cp_model

logger = logging.getLogger(__name__)


class BaseOrtoolsLinearSolver(BaseSolver):
    """
    OR-Tools의 pywraplp (LP, MIP) 솔버를 위한 기본 클래스.
    """

    def __init__(self, input_data, solver_name):
        super().__init__(input_data)
        self.solver = pywraplp.Solver.CreateSolver(solver_name)
        if not self.solver:
            raise Exception(f"{solver_name} Solver not available.")

    def solve(self):
        try:
            self._create_variables()
            self._add_constraints()
            self._set_objective_function()

            start_time = time.time()
            status = self.solver.Solve()
            processing_time = time.time() - start_time
            if settings.SAVE_MODEL_FILE:
                export_ortools_solver(self.solver, f'{self.problem_type}.mps')
            if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
                results = self._extract_results()
                error_msg = None
                if status == pywraplp.Solver.FEASIBLE:
                    logger.warning(f"Feasible solution found for {self.problem_type}.")
                return results, error_msg, processing_time
            else:
                # ... (에러 처리 로직)
                error_msg = f"Optimal solution not found. Solver status: {status}"
                return None, error_msg, processing_time

        except Exception as e:
            # BaseSolver의 에러 처리 로직을 그대로 활용
            return super().solve()


class BaseOrtoolsCpSolver(BaseSolver):
    """
    OR-Tools의 CP-SAT 솔버를 위한 기본 클래스.
    """

    def __init__(self, input_data):
        super().__init__(input_data)
        self.model = cp_model.CpModel()
        # CP-SAT에서는 solver 객체를 solve 직전에 생성합니다.

    def _extract_results(self, solver):
        # CP-SAT의 결과 추출은 solver 객체가 필요하므로, 인자를 받도록 재정의합니다.
        raise NotImplementedError

    def solve(self):
        try:
            self._create_variables()
            self._add_constraints()
            self._set_objective_function()

            solver = cp_model.CpSolver()
            # solver.parameters.max_time_in_seconds = 30.0 # 필요시 시간 제한 설정
            if settings.SAVE_MODEL_FILE:
                export_cp_model(self.model, f'{self.problem_type}.mps')
            start_time = time.time()
            status = solver.Solve(self.model)
            processing_time = time.time() - start_time

            if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
                results = self._extract_results(solver)  # solver 객체를 전달
                error_msg = None
                return results, error_msg, processing_time
            else:
                error_msg = f"Optimal solution not found. Solver status: {solver.StatusName(status)}"
                return None, error_msg, processing_time

        except Exception as e:
            return super().solve()
