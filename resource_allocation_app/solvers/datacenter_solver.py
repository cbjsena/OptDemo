import logging
from math import floor
from ortools.linear_solver import pywraplp
from common_utils.ortools_solvers import BaseOrtoolsLinearSolver

logger = logging.getLogger(__name__)


class DataCenterCapacitySolver(BaseOrtoolsLinearSolver):
    """
    데이터 센터 용량 계획 문제를 정의하고 해결하는 클래스.
    OR-Tools의 MIP 솔버(CBC)를 사용합니다.
    """

    def __init__(self, input_data):
        """
        생성자: BaseOrtoolsLinearSolver를 초기화하고 데이터 센터 문제에 특화된 데이터를 파싱합니다.
        """
        super().__init__(input_data, 'CBC')  # MIP를 지원하는 CBC 솔버 사용

        # 입력 데이터 파싱

        self.server_data = self.input_data.get('server_data')
        self.demand_data = self.input_data.get('demand_data')
        self.num_server_data = self.input_data.get('num_server_types')
        self.num_demand_data = self.input_data.get('num_services')
        self.global_constraints = self.input_data.get('global_constraints')
        self.total_power = self.global_constraints.get('total_power_kva')
        self.total_space = self.global_constraints.get('total_space_sqm')

        self.infinity = self.solver.infinity()
        self.all_server = range(self.num_server_data)
        self.all_demand = range(self.num_demand_data)
        # 결정 변수 초기화
        self.svr_var = []  # 구매할 서버 수량 (Sv_i)
        self.alloc_var = {}  # 서비스 할당량 (Alloc_si)

    def _create_variables(self):
        """결정 변수 (서버 구매 수량, 서비스 할당량)를 생성합니다."""
        logger.solve("--- 1. Creating Decision Variables ---")
        # Sv[i]: 서버 i 구매 수(정수 변수)
        self.svr_var = [self.solver.IntVar(0, self.infinity, f'Sv{i + 1}') for i in self.all_server]
        logger.solve(f"SV: 서버 i의 구매 개수, 총 {len(self.svr_var)}개 생성")
        for i, var in enumerate(self.svr_var):
            ub = floor(
                min(self.total_power / self.server_data[i].get('power_kva'), self.total_space / self.server_data[i].get('space_sqm')))
            logger.solve(f"  - {var.name()} (서버: {self.server_data[i].get('id', i)}), 범위: [{var.lb()}, {ub}]")

        # Dm[s]Sv[i]: 서비스 s를 위해 서버 i에 할당된 "자원 단위" 또는 "서비스 인스턴스 수"
        # 여기서는, 각 서비스가 특정 양의 CPU, RAM, Storage를 요구하고,
        # 각 서버이 특정 양의 CPU, RAM, Storage를 제공한다고 가정.

        for i_idx in self.all_server:
            for s_idx in self.all_demand:
                service = self.demand_data[s_idx]
                # 서비스 s의 최대 유닛 수 (수요) 만큼 변수 생성 고려
                # 또는, 총 제공 가능한 서비스 유닛을 변수로 할 수도 있음.f
                # 여기서는 서비스 s를 서버 i에서 몇 '유닛'만큼 제공할지를 변수로 설정.
                # 이 '유닛'은 해당 서비스의 요구 자원에 맞춰짐.
                # 서비스 s를 서버 i에서 몇 유닛 제공할지 (이산적인 서비스 유닛으로 가정)
                max_units_s = service.get('max_units', self.infinity) if service.get('max_units') is not None else self.infinity
                self.alloc_var[s_idx, i_idx] = self.solver.IntVar(0, max_units_s if max_units_s != self.infinity else self.infinity(),
                                                    f'Alloc{i_idx + 1}_{s_idx + 1}')

        logger.solve(f"Alloc_ij: 서버 i에 할당된 서비스 j의 용량, 총 {len(self.alloc_var)}개 생성")
        # 모든 변수를 출력하기는 너무 많을 수 있으므로, 일부만 예시로 출력하거나 요약
        if len(self.alloc_var) > 10:  # 변수가 많을 경우 일부만 출력
            logger.solve(
                f"  (예시) X_s{self.demand_data[0].get('id', 0)}_i{self.server_data[0].get('id', 0)}, X_s{self.demand_data[0].get('id', 0)}_i{self.server_data[1].get('id', 1)}, ...")
        else:
            for (s_idx, i_idx), var in self.alloc_var.items():
                logger.solve(
                    f"  - {var.name()} (서버: {self.server_data[i_idx].get('id', i_idx)}),서비스: {self.demand_data[s_idx].get('id', s_idx)},  범위: [{var.lb()}, {var.ub()}]")
        logger.solve(f"Created {len(self.svr_var)} Sv variables and {len(self.alloc_var)} Alloc variables.")

    def _add_constraints(self):
        """모델에 모든 제약 조건들을 추가합니다."""
        logger.solve("--- 2. Adding Constraints ---")
        self.infinity = self.solver.infinity()

        # 1. 총 예산, 전력, 공간 제약
        total_budget_constraint = self.solver.Constraint(0, self.global_constraints.get('total_budget', self.infinity), 'total_budget')
        total_power_constraint = self.solver.Constraint(0, self.global_constraints.get('total_power_kva', self.infinity),
                                                   'total_power')
        total_space_constraint = self.solver.Constraint(0, self.global_constraints.get('total_space_sqm', self.infinity),
                                                   'total_space')
        for i in self.all_server:
            total_budget_constraint.SetCoefficient(self.svr_var[i], self.server_data[i].get('cost', 0))
            total_power_constraint.SetCoefficient(self.svr_var[i], self.server_data[i].get('power_kva', 0))
            total_space_constraint.SetCoefficient(self.svr_var[i], self.server_data[i].get('space_sqm', 0))

        budget_terms = []
        power_terms = []
        space_terms = []
        for i in self.all_server:
            cost = self.server_data[i].get('cost', 0)
            power = self.server_data[i].get('power_kva', 0)
            space = self.server_data[i].get('space_sqm', 0)
            if cost != 0:
                budget_terms.append(f"{cost}*{self.svr_var[i].name()}")
            if power != 0:
                power_terms.append(f"{power}*{self.svr_var[i].name()}")
            if space != 0:
                space_terms.append(f"{space}*{self.svr_var[i].name()}")
        budget_expr_str = " + ".join(budget_terms)
        power_expr_str = " + ".join(power_terms)
        space_expr_str = " + ".join(space_terms)
        logger.solve(
            f"total_budget: {total_budget_constraint.lb()} <= {budget_expr_str} <= {total_budget_constraint.ub()}")
        logger.solve(
            f"total_power: {total_power_constraint.lb()} <= {power_expr_str} <= {total_power_constraint.ub()}")
        logger.solve(
            f"total_space: {total_space_constraint.lb()} <= {space_expr_str} <= {total_space_constraint.ub()}")

        # 4. 각 자원(CPU, RAM, Storage)에 대한 용량 제약
        # 각 서버 i가 제공하는 총 CPU = Ns[i] * self.server_data[i]['cpu_cores']
        # 각 서비스 s의 유닛이 요구하는 CPU = demands_data[s]['req_cpu_cores']
        # 총 요구 CPU = sum over s,i (X_si[s,i] * demands_data[s]['req_cpu_cores'])
        # 이는 잘못된 접근. X_si는 서비스 s를 서버 i에서 몇 유닛 제공하는지.
        # 서버 i에 할당된 서비스들의 총 요구 자원이 서버 i의 총 제공 자원을 넘을 수 없음.

        # 수정된 제약: 각 서버 i에 대해, 해당 서버에 할당된 모든 서비스의 자원 요구량 합계는
        # 해당 서버의 총 구매된 용량을 초과할 수 없음.
        resource_types = ['cpu_cores', 'ram_gb', 'storage_tb']
        for i_idx in self.all_server:  # 각 서버에 대해
            server_type = self.server_data[i_idx]
            for res_idx, resource in enumerate(resource_types):  # 각 자원 유형에 대해
                # # 서버 i가 제공하는 총 자원량
                # # Ns[i_idx] * server.get(resource, 0)
                # # 서버 i에 할당된 모든 서비스 유닛들이 소모하는 총 자원량
                # # sum (X_si[s_idx, i_idx] * demands_data[s_idx].get(f'req_{resource}', 0) for s_idx in range(num_services))
                # constraint_res = self.solver.Constraint(-self.infinity, 0, f'res_{resource}server{i_idx}')
                # # 제공량 (우변으로 넘기면 <= 0)
                # constraint_res.SetCoefficient(Ns[i_idx], -server.get(resource, 0))  # 제공량은 음수로
                # # 소비량 (좌변에 그대로)
                # for s_idx in range(num_services):
                #     if (s_idx, i_idx) in X_si:  # 해당 변수가 존재할 때만
                #         service = demands_data[s_idx]
                #         constraint_res.SetCoefficient(X_si[s_idx, i_idx], service.get(f'req_{resource}', 0))  # 소비량은 양수로
                # logger.solve(f"Added resource constraint for {resource} on server type {server.get('id', i_idx)}.")

                # 제약 조건을 생성하기 전에 정의된 변수가 존재하는지 확인
                coeffs = []
                terms = []
                # Ns[i_idx] * server.get(resource, 0) 만큼의 자원 제공
                coeffs.append(-server_type.get(resource, 0))
                terms.append(self.svr_var[i_idx])

                for s_idx in self.all_demand:
                    if (s_idx, i_idx) in self.alloc_var:
                        service = self.demand_data[s_idx]
                        req_resource = service.get(f'req_{resource}', 0)
                        coeffs.append(req_resource)
                        terms.append(self.alloc_var[s_idx, i_idx])

                # 모든 계수가 0이 아닌 경우에만 제약을 추가 (자원 요구사항이 없는 경우 불필요)
                if any(c != 0 for c in coeffs):
                    constraint_expr = self.solver.Sum(terms[j] * coeffs[j] for j in range(len(terms)))
                    constraint_name = f'con_{resource}_{server_type.get("id", i_idx)}'
                    # sum(X_si[s,i] * req_res[s]) <= Ns[i] * server_res[i] 형태로 표현 가능
                    # 즉, sum(X_si[s,i] * req_res[s]) - Ns[i] * server_res[i] <= 0
                    constraint = self.solver.Add(constraint_expr <= 0, constraint_name)
                    logger.solve(f"{constraint.name()}: {constraint_expr} <= 0")

        # 5. 각 서비스의 최대 수요(유닛) 제약 (선택 사항, X_si 변수 상한으로 이미 반영됨)
        # sum over i (X_si[s,i]) <= demands_data[s]['max_units'] (또는 == nếu 정확히 수요 충족)
        for s_idx in self.all_demand:
            service = self.demand_data[s_idx]
            max_units_s = service.get('max_units')
            if max_units_s is not None and max_units_s != self.infinity:
                # 서비스 s에 대해 모든 서버에서 제공되는 총 유닛 수는 max_units_s를 넘을 수 없음
                constraint_demand_s = self.solver.Constraint(0, max_units_s, f'demand_service_{s_idx}')
                for i_idx in self.all_server:
                    if (s_idx, i_idx) in self.alloc_var:
                        constraint_demand_s.SetCoefficient(self.alloc_var[s_idx, i_idx], 1)
                # logger.solve(f"service_{service.get('id', s_idx)}: sum(X_si[{service.get('id', s_idx)},i]) <= {max_units_s}")
                logger.solve(f"service_{service.get('id', s_idx)}: sum(Dm[{s_idx},i]) <= {max_units_s}")

    def _set_objective_function(self):
        """목표 함수 (총 이익 = 총 서비스 수익 - 총 서버 구매 비용)를 설정합니다."""
        logger.solve("--- 3. Setting Objective Function ---")
        objective = self.solver.Objective()
        # 서버 구매 비용 (음수)
        for i in self.all_server:
            objective.SetCoefficient(self.svr_var[i], -self.server_data[i].get('cost', 0))

        # 서비스 수익 (양수)
        for s_idx in self.all_demand:
            service = self.demand_data[s_idx]
            for i_idx in self.all_server:
                if (s_idx, i_idx) in self.alloc_var:
                    objective.SetCoefficient(self.alloc_var[s_idx, i_idx], service.get('revenue_per_unit', 0))

        objective.SetMaximization()
        logger.solve(f"\n**목표 함수:** 총 이익 극대화 (서비스 수익 - 서버 구매 비용)")
        logger.solve(f"  목표: Maximize sum(X_si * revenue_per_unit) - sum(Ns * cost)")

        obj_terms = []
        for i in self.all_server:
            coff = self.server_data[i].get('cost', 0)
            if coff != 0:
                obj_terms.append(f"{-coff}*{self.svr_var[i].name()}")

        for s_idx in self.all_demand:
            service = self.demand_data[s_idx]
            for i_idx in self.all_server:
                if (s_idx, i_idx) in self.alloc_var:
                    coff = service.get('revenue_per_unit', 0)
                    if coff != 0:
                        obj_terms.append(f"{coff}*{self.alloc_var[s_idx, i_idx].name()}")
        obj_expr_str = " + ".join(obj_terms)
        logger.solve(f"obj_exp: {obj_expr_str}")


    def _extract_results(self):
        """솔버 실행 후, 결과를 가공하여 딕셔너리 형태로 반환합니다."""
        logger.info("Extracting results for data center capacity...")

        # 구매된 서버 정보 추출
        purchased_servers = []
        total_server_cost = 0
        total_power_used = 0
        total_space_used = 0
        for i in self.all_server:
            count = int(self.svr_var[i].solution_value())
            if count > 0:
                server = self.server_data[i]
                cost = server.get('cost', 0)
                purchased_servers.append({
                    'type_id': server.get('id'),
                    'count': count,
                    'unit_cost': cost,
                    'total_cost_for_type': count * cost
                })
                total_server_cost += count * cost
                total_power_used += count * server.get('power_kva', 0)
                total_space_used += count * server.get('space_sqm', 0)

        # 할당된 서비스 정보 추출
        service_allocations = []
        total_service_revenue = 0
        for s in self.all_demand:
            service = self.demand_data[s]
            total_units_provided = 0
            allocations_detail = []
            for i in self.all_server:
                units = int(self.alloc_var[(s, i)].solution_value())
                if units > 0:
                    total_units_provided += units
                    allocations_detail.append({
                        'server_type_id': self.server_data[i].get('id'),
                        'units_allocated': units
                    })

            if total_units_provided > 0:
                revenue = total_units_provided * service.get('revenue_per_unit', 0)
                service_allocations.append({
                    'service_id': service.get('id'),
                    'total_units_provided': total_units_provided,
                    'revenue_from_service': revenue,
                    'allocations': allocations_detail
                })
                total_service_revenue += revenue

        return {
            'purchased_servers': purchased_servers,
            'service_allocations': service_allocations,
            'total_profit': self.solver.Objective().Value(),
            'total_server_cost': total_server_cost,
            'total_service_revenue': total_service_revenue,
            'total_power_used': total_power_used,
            'total_space_used': total_space_used,
        }
