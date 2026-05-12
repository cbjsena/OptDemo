import logging
from gurobipy import GRB
from common_utils.gurobi_solvers import BaseGurobiSolver
from common_utils.ortools_solvers import BaseOrtoolsLinearSolver

logger = logging.getLogger(__name__)


class _AdvancedSolverMixin:
    """
    Advanced Vessel Deployment 공통 초기화 및 결과 추출 로직.

    선형화 기법: V_r 이산 분해
    - x[r, s, v]: Lane r에 크기 s 선박 배치 수 (V_r = v일 때)
    - delta[r, v]: Lane r의 선박 수가 v인지 여부 (이진)
    """

    def _init_advanced_data(self):
        self.vessel_sizes = self.input_data.get('vessel_sizes', [])
        self.vessel_availability = self.input_data.get('vessel_availability', {})
        self.trades = self.input_data.get('trades', [])
        self.v_min = self.input_data.get('v_min', 3)
        self.v_max = self.input_data.get('v_max', 15)
        self.v_range = list(range(self.v_min, self.v_max + 1))

        self.candidate_routes = []
        self.trade_route_indices = {}
        idx = 0
        for trade in self.trades:
            code = trade['code']
            max_lanes = trade.get('max_lanes', 8)
            self.trade_route_indices[code] = []
            for i in range(max_lanes):
                self.candidate_routes.append({
                    'name': f"{code}{i + 1}",
                    'trade': code,
                })
                self.trade_route_indices[code].append(idx)
                idx += 1

        self.num_routes = len(self.candidate_routes)
        self.num_sizes = len(self.vessel_sizes)
        self.x = {}
        self.delta = {}

    def _extract_common_results(self, get_val, get_obj):
        """공통 결과 추출 로직"""
        logger.info("--- 4. Extracting Results ---")

        deployment_matrix = []
        total_vessels_used = 0

        for r in range(self.num_routes):
            route = self.candidate_routes[r]
            selected_v = 0
            for v in self.v_range:
                if get_val(self.delta[r, v]) > 0.5:
                    selected_v = v
                    break
            if selected_v == 0:
                continue

            row = {
                'name': route['name'],
                'trade': route['trade'],
                'V_r': selected_v,
                'deployment': [],
                'total_vessels': 0,
                'total_capacity': 0,
                'transport': 0,
            }
            for s in range(self.num_sizes):
                count = int(round(get_val(self.x[r, s, selected_v])))
                row['deployment'].append(count)
                row['total_vessels'] += count
                row['total_capacity'] += count * self.vessel_sizes[s]

            row['transport'] = round(row['total_capacity'] / selected_v) if selected_v > 0 else 0
            total_vessels_used += row['total_vessels']
            deployment_matrix.append(row)

        fleet_usage = []
        for s in range(self.num_sizes):
            size = self.vessel_sizes[s]
            used = sum(
                int(round(get_val(self.x[r, s, v])))
                for r in range(self.num_routes)
                for v in self.v_range
            )
            avail = self.vessel_availability.get(str(size), 0)
            fleet_usage.append({
                'size': size,
                'used': used,
                'available': avail,
                'utilization': round(used / avail * 100, 1) if avail > 0 else 0,
            })

        trade_summary = []
        for trade in self.trades:
            code = trade['code']
            trade_rows = [row for row in deployment_matrix if row['trade'] == code]
            vessels = sum(row['total_vessels'] for row in trade_rows)
            capacity = sum(row['total_capacity'] for row in trade_rows)
            transport = sum(row['transport'] for row in trade_rows)
            active_lanes = len(trade_rows)
            trade_summary.append({
                'code': code,
                'demand': trade['demand'],
                'max_lanes': trade.get('max_lanes', 8),
                'active_lanes': active_lanes,
                'vessels': vessels,
                'capacity': capacity,
                'transport': transport,
                'surplus': transport - trade['demand'],
            })

        results = {
            'deployment_matrix': deployment_matrix,
            'fleet_usage': fleet_usage,
            'trade_summary': trade_summary,
            'total_vessels_used': total_vessels_used,
            'vessel_sizes': self.vessel_sizes,
            'objective_value': get_obj(),
            'v_min': self.v_min,
            'v_max': self.v_max,
        }

        logger.info(f"Total vessels deployed: {total_vessels_used}")
        logger.info(f"Active lanes: {len(deployment_matrix)} / {self.num_routes} candidates")
        logger.info(f"Objective value: {get_obj()}")

        return results


# ============================================================
# Gurobi 솔버
# ============================================================
class VesselDeploymentAdvancedSolver(_AdvancedSolverMixin, BaseGurobiSolver):
    """Advanced Vessel Deployment - Gurobi 솔버"""

    def __init__(self, input_data):
        BaseGurobiSolver.__init__(self, input_data)
        self._init_advanced_data()

        # Gurobi 파라미터 설정
        self.model.Params.MIPGap = 0.02
        self.model.Params.TimeLimit = 120
        self.model.Params.Threads = 0
        self.model.Params.OutputFlag = 1

    def _create_variables(self):
        logger.info("--- 1. Creating Decision Variables (Gurobi) ---")
        for r in range(self.num_routes):
            route = self.candidate_routes[r]
            for s in range(self.num_sizes):
                size = self.vessel_sizes[s]
                avail = self.vessel_availability.get(str(size), 0)
                for v in self.v_range:
                    upper = min(avail, v)
                    self.x[r, s, v] = self.model.addVar(
                        vtype=GRB.INTEGER, lb=0, ub=upper,
                        name=f'x_{route["name"]}_{size}_v{v}'
                    )
            for v in self.v_range:
                self.delta[r, v] = self.model.addVar(
                    vtype=GRB.BINARY,
                    name=f'delta_{route["name"]}_v{v}'
                )
        self.model.update()
        logger.info(f"Created {len(self.x)} integer + {len(self.delta)} binary variables")

    def _add_constraints(self):
        logger.info("--- 2. Adding Constraints (Gurobi) ---")

        for r in range(self.num_routes):
            self.model.addConstr(
                sum(self.delta[r, v] for v in self.v_range) <= 1,
                name=f'lane_act_{r}'
            )

        for r in range(self.num_routes):
            for v in self.v_range:
                self.model.addConstr(
                    sum(self.x[r, s, v] for s in range(self.num_sizes)) == v * self.delta[r, v],
                    name=f'link_{r}_v{v}'
                )

        for trade in self.trades:
            code = trade['code']
            demand = trade['demand']
            route_indices = self.trade_route_indices[code]
            expr = sum(
                self.x[r, s, v] * (self.vessel_sizes[s] / v)
                for r in route_indices for v in self.v_range for s in range(self.num_sizes)
            )
            self.model.addConstr(expr >= demand, name=f'demand_{code}')

        for s in range(self.num_sizes):
            size = self.vessel_sizes[s]
            avail = self.vessel_availability.get(str(size), 0)
            expr = sum(self.x[r, s, v] for r in range(self.num_routes) for v in self.v_range)
            self.model.addConstr(expr <= avail, name=f'fleet_{size}')

    def _set_objective_function(self):
        logger.info("--- 3. Setting Objective (Gurobi) ---")
        total = sum(
            self.x[r, s, v]
            for r in range(self.num_routes) for s in range(self.num_sizes) for v in self.v_range
        )
        self.model.setObjective(total, GRB.MINIMIZE)

    def _extract_results(self):
        return self._extract_common_results(
            get_val=lambda var: var.X,
            get_obj=lambda: self.model.ObjVal,
        )


# ============================================================
# OR-Tools 솔버
# ============================================================
class VesselDeploymentAdvancedOrtoolsSolver(_AdvancedSolverMixin, BaseOrtoolsLinearSolver):
    """Advanced Vessel Deployment - OR-Tools SAT (CP-SAT) 솔버"""

    def __init__(self, input_data):
        BaseOrtoolsLinearSolver.__init__(self, input_data, 'SAT')
        self._init_advanced_data()

    def _create_variables(self):
        logger.info("--- 1. Creating Decision Variables (OR-Tools SAT) ---")
        for r in range(self.num_routes):
            route = self.candidate_routes[r]
            for s in range(self.num_sizes):
                size = self.vessel_sizes[s]
                avail = self.vessel_availability.get(str(size), 0)
                for v in self.v_range:
                    upper = min(avail, v)
                    self.x[r, s, v] = self.solver.IntVar(
                        0, upper, f'x_{route["name"]}_{size}_v{v}'
                    )
            for v in self.v_range:
                self.delta[r, v] = self.solver.BoolVar(
                    f'delta_{route["name"]}_v{v}'
                )
        logger.info(f"Created {len(self.x)} integer + {len(self.delta)} binary variables")

    def _add_constraints(self):
        logger.info("--- 2. Adding Constraints (OR-Tools SAT) ---")

        for r in range(self.num_routes):
            self.solver.Add(sum(self.delta[r, v] for v in self.v_range) <= 1)

        for r in range(self.num_routes):
            for v in self.v_range:
                self.solver.Add(
                    sum(self.x[r, s, v] for s in range(self.num_sizes)) == v * self.delta[r, v]
                )

        for trade in self.trades:
            code = trade['code']
            demand = trade['demand']
            route_indices = self.trade_route_indices[code]
            expr = sum(
                self.x[r, s, v] * (self.vessel_sizes[s] / v)
                for r in route_indices for v in self.v_range for s in range(self.num_sizes)
            )
            self.solver.Add(expr >= demand)

        for s in range(self.num_sizes):
            size = self.vessel_sizes[s]
            avail = self.vessel_availability.get(str(size), 0)
            expr = sum(self.x[r, s, v] for r in range(self.num_routes) for v in self.v_range)
            self.solver.Add(expr <= avail)

    def _set_objective_function(self):
        logger.info("--- 3. Setting Objective (OR-Tools SAT) ---")
        total = sum(
            self.x[r, s, v]
            for r in range(self.num_routes) for s in range(self.num_sizes) for v in self.v_range
        )
        self.solver.Minimize(total)
        # SAT 시간 제한 설정
        self.solver.SetTimeLimit(120_000)  # 최대 120초

    def _extract_results(self):
        return self._extract_common_results(
            get_val=lambda var: var.solution_value(),
            get_obj=lambda: self.solver.Objective().Value(),
        )
