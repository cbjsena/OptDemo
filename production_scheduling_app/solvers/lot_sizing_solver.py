import logging
from ortools.linear_solver import pywraplp
from common_utils.ortools_solvers import BaseOrtoolsLinearSolver
from core.decorators import log_solver_make

logger = logging.getLogger('production_scheduling_app')


class LotSizingSolver(BaseOrtoolsLinearSolver):
    """
    Lot Sizing 문제를 정의하고 해결하는 클래스.
    """

    def __init__(self, input_data):
        """
        생성자: BaseOrtoolsLinearSolver를 초기화하고 Lot Sizing 문제에 특화된 데이터를 파싱합니다.
        """
        super().__init__(input_data, 'CBC')  # MIP 솔버인 CBC 사용

        # --- 1. 데이터 파싱 및 가공 ---
        self.demands = self.input_data.get('demands', [])
        self.num_periods = len(self.demands)
        if self.num_periods == 0:
            raise ValueError("수요 데이터가 없습니다.")

        self.setup_costs = self.input_data.get('setup_costs')
        self.prod_costs = self.input_data.get('prod_costs')
        self.holding_costs = self.input_data.get('holding_costs')
        self.capacities = self.input_data.get('capacities')

        # 모델링에 사용할 변수 초기화
        self.x = []  # 생산량
        self.y = []  # 생산 여부
        self.I = []  # 재고량

    @log_solver_make
    def _create_variables(self):
        """결정 변수 (생산량, 생산 여부, 재고량)를 생성합니다."""
        infinity = self.solver.infinity()
        self.x = [self.solver.IntVar(0, infinity, f'x_{t}') for t in range(self.num_periods)]
        self.y = [self.solver.BoolVar(f'y_{t}') for t in range(self.num_periods)]
        self.I = [self.solver.IntVar(0, infinity, f'I_{t}') for t in range(self.num_periods)]

    @log_solver_make
    def _add_constraints(self):
        """재고 균형 및 생산 용량 제약을 추가합니다."""
        # 제약 1: 재고 균형 제약
        for t in range(self.num_periods):
            previous_inventory = self.I[t - 1] if t > 0 else 0
            self.solver.Add(previous_inventory + self.x[t] == self.demands[t] + self.I[t], f"inventory_balance_{t}")

        # 제약 2: 생산 용량 제약
        for t in range(self.num_periods):
            self.solver.Add(self.x[t] <= self.capacities[t] * self.y[t], f"capacity_constraint_{t}")

    @log_solver_make
    def _set_objective_function(self):
        """총비용(생산, 재고, 셋업) 최소화를 목표 함수로 설정합니다."""
        objective = self.solver.Objective()
        for t in range(self.num_periods):
            objective.SetCoefficient(self.y[t], self.setup_costs[t])
            objective.SetCoefficient(self.x[t], self.prod_costs[t])
            objective.SetCoefficient(self.I[t], self.holding_costs[t])
        objective.SetMinimization()

    def _extract_results(self):
        """솔버 실행 후, 최적 생산 계획과 비용 내역을 추출합니다."""
        schedule = []
        total_setup_cost = 0
        total_prod_cost = 0
        total_holding_cost = 0

        for t in range(self.num_periods):
            is_setup = self.y[t].solution_value() > 0.5
            production_amount = self.x[t].solution_value()
            inventory_level = self.I[t].solution_value()

            schedule.append({
                'period': t + 1,
                'demand': self.demands[t],
                'production_amount': round(production_amount),
                'inventory_level': round(inventory_level),
                'is_setup': bool(is_setup)
            })

            if is_setup:
                total_setup_cost += self.setup_costs[t]
            total_prod_cost += self.prod_costs[t] * production_amount
            total_holding_cost += self.holding_costs[t] * inventory_level

        return {
            'schedule': schedule,
            'total_cost': self.solver.Objective().Value(),
            'cost_breakdown': {
                'setup': round(total_setup_cost),
                'production': round(total_prod_cost),
                'holding': round(total_holding_cost),
            }
        }
