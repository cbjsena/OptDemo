from ortools.linear_solver import pywraplp  # OR-Tools MIP solver (실제로는 LP 솔버 사용)
from ortools.sat.python import cp_model

import logging

logger = logging.getLogger(__name__)


def run_diet_optimizer(input_data):
    logger.info("Running Diet Problem Optimizer.")

    foods = input_data['food_items']
    nutrients = input_data['nutrient_reqs']
    num_foods = len(foods)
    num_nutrients = len(nutrients)

    solver = pywraplp.Solver.CreateSolver('GLOP')
    if not solver:
        return None, "오류: 선형 계획법 솔버(GLOP)를 생성할 수 없습니다.", 0.0

    # 변수 x_i: i번째 음식의 섭취량
    x = [solver.NumVar(f['min_intake'], f['max_intake'], f['name']) for f in foods]
    logger.debug(f"Created {len(x)} food variables.")

    # 제약: 각 영양소의 최소/최대 섭취량 만족
    for i in range(num_nutrients):
        constraint = solver.Constraint(nutrients[i]['min'], nutrients[i]['max'], nutrients[i]['name'])
        for j in range(num_foods):
            constraint.SetCoefficient(x[j], foods[j]['nutrients'][i])
    logger.debug(f"Added {num_nutrients} nutrient constraints.")

    # 목표 함수: 총 비용 최소화
    objective = solver.Objective()
    for i in range(num_foods):
        objective.SetCoefficient(x[i], foods[i]['cost'])
    objective.SetMinimization()
    logger.debug("Objective set to minimize total cost.")

    # 해결
    status = solver.Solve()
    logger.info(f"Solver status: {status}, Time: {solver.WallTime():.2f} ms")

    # 결과 추출
    results = {'diet_plan': [], 'total_cost': 0, 'nutrient_summary': []}
    error_msg = None

    if status == pywraplp.Solver.OPTIMAL:
        results['total_cost'] = solver.Objective().Value()

        for i in range(num_foods):
            intake = x[i].solution_value()
            if intake > 1e-6:  # 매우 작은 값은 무시
                results['diet_plan'].append({
                    'name': foods[i]['name'],
                    'intake': round(intake, 2),
                    'cost': round(intake * foods[i]['cost'], 2)
                })

        for i in range(num_nutrients):
            total_nutrient_intake = sum(foods[j]['nutrients'][i] * x[j].solution_value() for j in range(num_foods))
            results['nutrient_summary'].append({
                'name': nutrients[i]['name'],
                'min_req': nutrients[i]['min'],
                'max_req': nutrients[i]['max'],
                'actual_intake': round(total_nutrient_intake, 2)
            })
    else:
        error_msg = "최적 식단을 찾지 못했습니다. 제약 조건이 너무 엄격하거나(INFEASIBLE), 문제가 잘못 정의되었을 수 있습니다."

    return results, error_msg, get_solving_time_sec(solver.WallTime())


def calculate_manual_diet_result(input_data, manual_quantities):
    """사용자가 입력한 수동 식단의 비용과 영양 성분을 계산합니다."""
    foods = input_data.get('foods', [])
    nutrients = input_data.get('nutrients', [])
    num_foods = len(foods)
    num_nutrients = len(nutrients)

    manual_results = {'diet_plan': [], 'total_cost': 0, 'nutrient_summary': []}
    total_cost = 0

    for j in range(num_foods):
        quantity = manual_quantities[j]
        total_cost += foods[j]['cost'] * quantity
        if quantity > 0:
            manual_results['diet_plan'].append({'name': foods[j]['name'], 'quantity': quantity})
    manual_results['total_cost'] = round(total_cost, 2)

    for i in range(num_nutrients):
        total_nutrient = sum(foods[j]['nutrients'][i] * manual_quantities[j] for j in range(num_foods))
        is_ok = nutrients[i]['min'] <= total_nutrient <= nutrients[i]['max']
        manual_results['nutrient_summary'].append({
            'name': nutrients[i]['name'],
            'min': nutrients[i]['min'],
            'max': nutrients[i]['max'],
            'total': round(total_nutrient, 2),
            'is_ok': is_ok
        })

    return manual_results


def run_sports_scheduling_optimizer(input_data):
    schedule_type = input_data.get('schedule_type')
    objective_choice = input_data.get('objective_choice')
    teams = input_data.get('teams', [])
    distance_matrix = input_data.get('distance_matrix')
    max_consecutive = input_data.get('max_consecutive', 3)
    num_teams_original = len(teams)

    logger.info(
        f"Running {schedule_type.upper()} Round-Robin Scheduler. Objective: {objective_choice}, Teams: {num_teams_original}")

    if num_teams_original < 2:
        return None, "오류: 최소 2개 팀이 필요합니다.", 0.0

    has_bye = False
    if num_teams_original % 2 != 0 and schedule_type == 'single':
        teams.append('BYE')
        has_bye = True
        logger.info("Odd number of teams for single round-robin. Added a BYE team.")

    num_teams = len(teams)

    # 리그 방식에 따라 슬롯(주차) 수 결정
    if schedule_type == 'single':
        num_slots = num_teams - 1
    else:  # double
        num_slots = 2 * (num_teams - 1)

    model = cp_model.CpModel()

    # --- 1. 결정 변수 ---
    # plays[s, h, a]: 시간 s에 홈팀 h가 원정팀 a와 경기하면 1
    plays = {}
    for s in range(num_slots):
        for h in range(num_teams):
            for a in range(num_teams):
                if h != a:
                    plays[(s, h, a)] = model.NewBoolVar(f'plays_s{s}_h{h}_a{a}')

    # --- 2. 제약 조건 ---
    # 제약 1: 각 팀은 각 슬롯에서 정확히 한 경기만
    for s in range(num_slots):
        for t in range(num_teams):
            home_games = [plays.get((s, t, a), 0) for a in range(num_teams) if t != a]
            away_games = [plays.get((s, h, t), 0) for h in range(num_teams) if t != h]
            model.AddExactlyOne(home_games + away_games)

    # 제약 2: 리그 방식에 따른 경기 수 제약
    for h in range(num_teams):
        for a in range(h + 1, num_teams):
            if schedule_type == 'single':
                # 싱글 라운드 로빈: 시즌 전체에 걸쳐 두 팀은 정확히 한 번 만남 (홈/원정 무관)
                model.Add(sum(plays.get((s, h, a), 0) + plays.get((s, a, h), 0) for s in range(num_slots)) == 1)
            else:  # double
                # 더블 라운드 로빈: 각 팀의 홈에서 정확히 한 번씩 만남
                model.Add(sum(plays.get((s, h, a), 0) for s in range(num_slots)) == 1)
                model.Add(sum(plays.get((s, a, h), 0) for s in range(num_slots)) == 1)

    # (선택적 제약) 제약 3: 같은 팀과 연속으로 경기하지 않음
    for h in range(num_teams):
        for a in range(num_teams):
            if h != a:
                for s in range(num_slots - 1):
                    # s주차와 s+1주차에 연속으로 같은 대진이 없도록 함
                    match_s = plays.get((s, h, a), 0) + plays.get((s, a, h), 0)
                    match_s_plus_1 = plays.get((s + 1, h, a), 0) + plays.get((s + 1, a, h), 0)
                    model.Add(match_s + match_s_plus_1 <= 1)

    for t in range(num_teams_original):
        for s in range(num_slots - max_consecutive):
            # N+1 경기 동안 홈 경기가 최소 1번 있어야 함 (N번 초과 연속 원정 방지)
            away_games_in_window = [plays.get((i, h, t), 0) for i in range(s, s + max_consecutive + 1) for h in
                                    range(num_teams) if t != h]
            model.Add(sum(away_games_in_window) <= max_consecutive)

            # N+1 경기 동안 원정 경기가 최소 1번 있어야 함 (N번 초과 연속 홈 방지)
            home_games_in_window = [plays.get((i, t, a), 0) for i in range(s, s + max_consecutive + 1) for a in
                                    range(num_teams) if t != a]
            model.Add(sum(home_games_in_window) <= max_consecutive)

    # 제약 3: 팀별 이동 거리 변수 생성
    team_travel_vars = [model.NewIntVar(0, 10000000, f'travel_{i}') for i in range(num_teams)]

    for t_idx in range(num_teams):
        travel_dist = []
        for s in range(num_slots):
            for opponent_idx in range(num_teams):
                if t_idx == opponent_idx:
                    continue
                # t_idx팀이 원정(away)일 때의 이동 거리: opponent_idx -> t_idx (이전 위치를 고려해야 더 정확함)
                # 단순화된 모델: 원정 경기 시, 자신의 홈에서 상대방 홈으로 이동한다고 가정
                # t_idx가 원정팀(a)으로 opponent_idx(h)와 경기할 때의 이동 거리
                travel_dist.append(distance_matrix[t_idx][opponent_idx] * plays[(s, opponent_idx, t_idx)])
        model.Add(team_travel_vars[t_idx] == sum(travel_dist))

    # --- 3. 목표 함수 설정 ---
    if objective_choice == 'minimize_travel':
        logger.debug("Objective set to: Minimize Total Travel Distance.")
        model.Minimize(sum(team_travel_vars))
        logger.debug("Objective set to: Minimize Total Travel Distance.")
    elif objective_choice == 'distance_gap':
        min_travel = model.NewIntVar(0, 10000000, 'min_travel')
        max_travel = model.NewIntVar(0, 10000000, 'max_travel')
        model.AddMinEquality(min_travel, team_travel_vars)
        model.AddMaxEquality(max_travel, team_travel_vars)
        model.Minimize(max_travel - min_travel)
        logger.debug("Objective set to: Minimize gap in travel distance (Distance Gap).")
    else:
        # 기본 목표: 특별한 목표 없음 (실행 가능한 해 찾기)
        logger.debug("Objective set to: Find a feasible solution.")

    # --- 4. 문제 해결 ---
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30.0
    logger.info("Solving the Sports Scheduling model...")
    status = solver.Solve(model)
    logger.info(f"Solver status: {status}, Time: {solver.WallTime():.2f} sec")

    # --- 5. 결과 추출 ---
    results = {'schedule': [], 'has_bye': has_bye, 'total_distance': 'N/A', 'team_distances': []}
    error_msg = None

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        schedule = []
        total_dist_calc = 0
        for s in range(num_slots):
            weekly_matchups = []
            for t1 in range(num_teams):
                for t2 in range(num_teams):
                    if (s, t1, t2) in plays and solver.Value(plays[(s, t1, t2)]) == 1:
                        # BYE 팀이 포함된 경기는 '휴식'으로 표시
                        if teams[t1] == 'BYE':
                            weekly_matchups.append((teams[t2], 'BYE'))
                        elif teams[t2] == 'BYE':
                            weekly_matchups.append((teams[t1], 'BYE'))
                        else:
                            weekly_matchups.append((teams[t1], teams[t2]))
            schedule.append({'week': s + 1, 'matchups': weekly_matchups})
        results['schedule'] = schedule
        for key, var in plays.items():
            if solver.Value(var) == 1:
                logger.solve(var.Name())
        team_distances_calc = []
        for i in range(num_teams):
            dist_val = solver.Value(team_travel_vars[i])
            team_distances_calc.append({'name': input_data['teams'][i], 'distance': round(dist_val)})
            total_dist_calc += dist_val

        results['total_distance'] = round(total_dist_calc)
        results['team_distances'] = sorted(team_distances_calc, key=lambda x: x['name'])
        results['team_distances'].sort(key=lambda x: x['name'])  # 팀 이름순 정렬
        dist = [item['distance'] for item in results['team_distances']]
        results['distance_gap'] = max(dist) - min(dist)
    else:
        error_msg = f"최적 스케줄을 찾지 못했습니다. (솔버 상태: {solver.StatusName(status)})"
        logger.error(f"Sports Scheduling failed: {error_msg}")

    return results, error_msg, get_solving_time_cp_sec(solver.WallTime())


def get_solving_time_sec(processing_time):
    # solver.WallTime(): if solver is CP-SAT then, sec else ms
    processing_time = processing_time / 1000
    return f"{processing_time:.3f}" if processing_time is not None else "N/A"


def get_solving_time_cp_sec(processing_time):
    # solver.WallTime(): if solver is CP-SAT then, sec else ms
    processing_time = processing_time
    return f"{processing_time:.3f}" if processing_time is not None else "N/A"