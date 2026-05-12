from django.shortcuts import render

from core.decorators import log_view_activity
from .solvers.palletizing_solver import PalletizingLogicSolver, PalletizingSolver
from .solvers.vessel_deployment_solver import VesselDeploymentSolver, VesselDeploymentLaneSolver
from .solvers.vessel_deployment_advanced_solver import (
    VesselDeploymentAdvancedSolver, VesselDeploymentAdvancedOrtoolsSolver
)


DEFAULT_PALLET = {
    'l': 100.0,
    'w': 100.0,
    'h': 100.0,
    'max_weight': 1200.0,
}

DEFAULT_BOX_TYPES = [
    {'id': 'BX1', 'l': 40.0, 'w': 30.0, 'h': 20.0, 'weight': 12.0, 'qty': 8, 'rotatable': True},
    {'id': 'BX2', 'l': 50.0, 'w': 40.0, 'h': 25.0, 'weight': 18.0, 'qty': 5, 'rotatable': True},
    {'id': 'BX3', 'l': 30.0, 'w': 20.0, 'h': 15.0, 'weight': 30.0, 'qty': 10, 'rotatable': False},
    {'id': 'BX4', 'l': 60.0, 'w': 40.0, 'h': 35.0, 'weight': 24.0, 'qty': 3, 'rotatable': True},
    {'id': 'BX5', 'l': 45.0, 'w': 35.0, 'h': 20.0, 'weight': 11.0, 'qty': 6, 'rotatable': True},
]

# Vessel Deployment 기본 데이터 (엑셀 기반)
DEFAULT_VESSEL_SIZES = [21000, 18000, 15000, 12000, 10000, 8000, 6000, 5000]

DEFAULT_VESSEL_AVAILABILITY = {
    '21000': 20, '18000': 6, '15000': 18, '12000': 58,
    '10000': 47, '8000': 19, '6000': 9, '5000': 8,
}

# Trade 기본 설정: code, 설명, Lane 수, 수요(TEU), Lane별 선박 수, Lane별 수요
DEFAULT_TRADES = [
    {'code': 'FE', 'desc': '극동 (Far East)', 'num_routes': 4, 'demand': 64000,
     'lane_vessels': [15, 14, 15, 14],
     'lane_demands': [18000, 16000, 16000, 14000]},
    {'code': 'MD', 'desc': '지중해 (Mediterranean)', 'num_routes': 2, 'demand': 19000,
     'lane_vessels': [14, 13],
     'lane_demands': [10000, 9000]},
    {'code': 'PS', 'desc': '태평양 남부 (Pacific South)', 'num_routes': 5, 'demand': 57000,
     'lane_vessels': [6, 6, 6, 6, 7],
     'lane_demands': [12000, 12000, 11000, 11000, 11000]},
    {'code': 'PN', 'desc': '태평양 북부 (Pacific North)', 'num_routes': 2, 'demand': 21000,
     'lane_vessels': [6, 7],
     'lane_demands': [10000, 11000]},
    {'code': 'EC', 'desc': '동안 (East Coast)', 'num_routes': 3, 'demand': 31000,
     'lane_vessels': [13, 14, 12],
     'lane_demands': [11000, 11000, 9000]},
    {'code': 'ME', 'desc': '중동 (Middle East)', 'num_routes': 2, 'demand': 10000,
     'lane_vessels': [8, 9],
     'lane_demands': [5000, 5000]},
]


@log_view_activity
def complex_app_introduction_view(request):
    context = {
        'active_model': 'Complex Optimization',
        'active_submenu': 'main_introduction',
    }
    return render(request, 'complex_app/complex_app_introduction.html', context)


@log_view_activity
def palletizing_introduction_view(request):
    context = {
        'active_model': 'Complex Optimization',
        'active_submenu': 'palletizing_introduction',
    }
    return render(request, 'complex_app/palletizing_introduction.html', context)


@log_view_activity
def lsnd_introduction_view(request):
    context = {
        'active_model': 'Complex Optimization',
        'active_submenu': 'lsnd_introduction',
    }
    return render(request, 'complex_app/lsnd_introduction.html', context)


@log_view_activity
def lsnd_advanced_model_view(request):
    context = {
        'active_model': 'Complex Optimization',
        'active_submenu': 'lsnd_advanced_model',
    }
    return render(request, 'complex_app/lsnd_advanced_model.html', context)


@log_view_activity
def lsnd_benchmark_data_view(request):
    context = {
        'active_model': 'Complex Optimization',
        'active_submenu': 'lsnd_benchmark_data',
    }
    return render(request, 'complex_app/lsnd_benchmark_data.html', context)


@log_view_activity
def vessel_deployment_introduction_view(request):
    context = {
        'active_model': 'Complex Optimization',
        'active_submenu': 'vessel_deployment_introduction',
    }
    return render(request, 'complex_app/vessel_deployment_introduction.html', context)


@log_view_activity
def vessel_deployment_demo_view(request):
    """Vessel Deployment 데모 뷰 - Trade별 수요 + Lane별 선박 수 입력"""
    source = request.POST if request.method == 'POST' else request.GET
    vessel_sizes = list(DEFAULT_VESSEL_SIZES)

    # 가용 수량 (POST에서 읽거나 기본값)
    vessel_availability = {}
    for size in vessel_sizes:
        key = f'avail_{size}'
        vessel_availability[str(size)] = int(source.get(key, DEFAULT_VESSEL_AVAILABILITY.get(str(size), 0)))

    # Trade 데이터 구성
    trades_data = []
    for t_idx, default_trade in enumerate(DEFAULT_TRADES):
        code = default_trade['code']
        desc = default_trade['desc']
        num_routes = default_trade['num_routes']
        demand = int(source.get(f'trade_demand_{t_idx}', default_trade['demand']))

        # Lane별 선박 수
        lane_vessels = []
        lane_names = []
        for l_idx in range(num_routes):
            lane_name = f"{code}{l_idx + 1}"
            lane_names.append(lane_name)
            v_count = int(source.get(f'lane_vessels_{t_idx}_{l_idx}', default_trade['lane_vessels'][l_idx]))
            lane_vessels.append(v_count)

        trades_data.append({
            'code': code,
            'desc': desc,
            'num_routes': num_routes,
            'demand': demand,
            'lane_vessels': lane_vessels,
            'lane_names': lane_names,
        })

    # 기존 총 선박 수 = 모든 Lane의 선박 수 합
    original_total_vessels = sum(v for t in trades_data for v in t['lane_vessels'])

    context = {
        'active_model': 'Complex Optimization',
        'active_submenu': 'vessel_deployment_demo',
        'vessel_sizes': vessel_sizes,
        'vessel_availability': vessel_availability,
        'trades_data': trades_data,
        'original_total_vessels': original_total_vessels,
        'results': None,
        'error_message': None,
        'success_message': None,
        'processing_time_seconds': "N/A",
        'vessel_saving': 0,
    }

    if request.method == 'POST':
        try:
            # 솔버 입력 데이터 구성
            solver_trades = []
            for trade in trades_data:
                solver_trades.append({
                    'code': trade['code'],
                    'num_routes': trade['num_routes'],
                    'demand': trade['demand'],
                    'lane_vessels': trade['lane_vessels'],
                })

            input_data = {
                'problem_type': 'vessel_deployment',
                'vessel_sizes': vessel_sizes,
                'vessel_availability': vessel_availability,
                'trades': solver_trades,
            }

            results, error_msg, processing_time = VesselDeploymentSolver(input_data).solve()
            context['processing_time_seconds'] = processing_time

            if error_msg:
                context['error_message'] = error_msg
            elif results:
                context['results'] = results
                context['vessel_saving'] = original_total_vessels - results['total_vessels_used']
                context['success_message'] = (
                    f"최적화 완료: 총 선박 {results['total_vessels_used']}척 "
                    f"(기존 {original_total_vessels}척 대비 "
                    f"{original_total_vessels - results['total_vessels_used']}척 절감, "
                    f"소요시간: {processing_time}초)"
                )
            else:
                context['error_message'] = "최적화 결과를 가져오지 못했습니다."

        except ValueError as ve:
            context['error_message'] = f"입력값 오류: {ve}"
        except Exception as e:
            context['error_message'] = f"처리 중 오류 발생: {e}"

    return render(request, 'complex_app/vessel_deployment_demo.html', context)


@log_view_activity
def vessel_deployment_demo2_view(request):
    """Vessel Deployment Demo2 - Lane별 수요 + Trade별 수요 충족"""
    source = request.POST if request.method == 'POST' else request.GET
    vessel_sizes = list(DEFAULT_VESSEL_SIZES)

    # 가용 수량
    vessel_availability = {}
    for size in vessel_sizes:
        key = f'avail_{size}'
        vessel_availability[str(size)] = int(source.get(key, DEFAULT_VESSEL_AVAILABILITY.get(str(size), 0)))

    # Trade 데이터 구성 (Lane별 수요 포함)
    trades_data = []
    for t_idx, default_trade in enumerate(DEFAULT_TRADES):
        code = default_trade['code']
        desc = default_trade['desc']
        num_routes = default_trade['num_routes']

        lane_vessels = []
        lane_demands = []
        lane_names = []
        for l_idx in range(num_routes):
            lane_name = f"{code}{l_idx + 1}"
            lane_names.append(lane_name)
            v_count = int(source.get(f'lane_vessels_{t_idx}_{l_idx}', default_trade['lane_vessels'][l_idx]))
            lane_vessels.append(v_count)
            d_lane = int(source.get(f'lane_demand_{t_idx}_{l_idx}', default_trade['lane_demands'][l_idx]))
            lane_demands.append(d_lane)

        # Trade 수요 = Lane 수요 합계 (자동 계산)
        trade_demand = sum(lane_demands)

        trades_data.append({
            'code': code,
            'desc': desc,
            'num_routes': num_routes,
            'demand': trade_demand,
            'lane_vessels': lane_vessels,
            'lane_demands': lane_demands,
            'lane_names': lane_names,
        })

    original_total_vessels = sum(v for t in trades_data for v in t['lane_vessels'])

    context = {
        'active_model': 'Complex Optimization',
        'active_submenu': 'vessel_deployment_demo2',
        'vessel_sizes': vessel_sizes,
        'vessel_availability': vessel_availability,
        'trades_data': trades_data,
        'original_total_vessels': original_total_vessels,
        'results': None,
        'error_message': None,
        'success_message': None,
        'processing_time_seconds': "N/A",
        'vessel_saving': 0,
    }

    if request.method == 'POST':
        try:
            solver_trades = []
            for trade in trades_data:
                solver_trades.append({
                    'code': trade['code'],
                    'num_routes': trade['num_routes'],
                    'demand': trade['demand'],
                    'lane_vessels': trade['lane_vessels'],
                    'lane_demands': trade['lane_demands'],
                })

            input_data = {
                'problem_type': 'vessel_deployment_lane',
                'vessel_sizes': vessel_sizes,
                'vessel_availability': vessel_availability,
                'trades': solver_trades,
            }

            results, error_msg, processing_time = VesselDeploymentLaneSolver(input_data).solve()
            context['processing_time_seconds'] = processing_time

            if error_msg:
                context['error_message'] = error_msg
            elif results:
                context['results'] = results
                context['vessel_saving'] = original_total_vessels - results['total_vessels_used']
                context['success_message'] = (
                    f"최적화 완료: 총 선박 {results['total_vessels_used']}척 "
                    f"(기존 {original_total_vessels}척 대비 "
                    f"{original_total_vessels - results['total_vessels_used']}척 절감, "
                    f"소요시간: {processing_time}초)"
                )
            else:
                context['error_message'] = "최적화 결과를 가져오지 못했습니다."

        except ValueError as ve:
            context['error_message'] = f"입력값 오류: {ve}"
        except Exception as e:
            context['error_message'] = f"처리 중 오류 발생: {e}"

    return render(request, 'complex_app/vessel_deployment_demo2.html', context)


@log_view_activity
def vessel_deployment_advanced_model_view(request):
    context = {
        'active_model': 'Complex Optimization',
        'active_submenu': 'vessel_deployment_advanced_model',
    }
    return render(request, 'complex_app/vessel_deployment_advanced_model.html', context)


# Demo3 기본 데이터: Trade별 최대 Lane 수
DEFAULT_TRADES_DEMO3 = [
    {'code': 'FE', 'desc': '극동 (Far East)', 'demand': 64000, 'max_lanes': 6},
    {'code': 'MD', 'desc': '지중해 (Mediterranean)', 'demand': 19000, 'max_lanes': 4},
    {'code': 'PS', 'desc': '태평양 남부 (Pacific South)', 'demand': 57000, 'max_lanes': 7},
    {'code': 'PN', 'desc': '태평양 북부 (Pacific North)', 'demand': 21000, 'max_lanes': 4},
    {'code': 'EC', 'desc': '동안 (East Coast)', 'demand': 31000, 'max_lanes': 5},
    {'code': 'ME', 'desc': '중동 (Middle East)', 'demand': 10000, 'max_lanes': 3},
]


@log_view_activity
def vessel_deployment_demo3_view(request):
    """Vessel Deployment Demo3 - Advanced: Lane 수 & V_r 최적화"""
    source = request.POST if request.method == 'POST' else request.GET
    vessel_sizes = list(DEFAULT_VESSEL_SIZES)

    # 가용 수량
    vessel_availability = {}
    for size in vessel_sizes:
        key = f'avail_{size}'
        vessel_availability[str(size)] = int(source.get(key, DEFAULT_VESSEL_AVAILABILITY.get(str(size), 0)))

    # 솔버 선택 (기본: OR-Tools SAT, Gurobi는 정식 라이선스 필요)
    solver_method = source.get('solver_method', 'ortools')

    # V_r 범위
    v_min = int(source.get('v_min', 3))
    v_max = int(source.get('v_max', 15))
    v_min = max(1, min(v_min, 30))
    v_max = max(v_min, min(v_max, 30))

    # Trade 데이터 구성
    trades_data = []
    for t_idx, default_trade in enumerate(DEFAULT_TRADES_DEMO3):
        code = default_trade['code']
        desc = default_trade['desc']
        demand = int(source.get(f'trade_demand_{t_idx}', default_trade['demand']))
        max_lanes = int(source.get(f'max_lanes_{t_idx}', default_trade['max_lanes']))
        max_lanes = max(1, min(max_lanes, 15))

        trades_data.append({
            'code': code,
            'desc': desc,
            'demand': demand,
            'max_lanes': max_lanes,
        })

    context = {
        'active_model': 'Complex Optimization',
        'active_submenu': 'vessel_deployment_demo3',
        'vessel_sizes': vessel_sizes,
        'vessel_availability': vessel_availability,
        'v_min': v_min,
        'v_max': v_max,
        'solver_method': solver_method,
        'trades_data': trades_data,
        'results': None,
        'error_message': None,
        'success_message': None,
        'processing_time_seconds': "N/A",
    }

    if request.method == 'POST':
        try:
            solver_trades = []
            for trade in trades_data:
                solver_trades.append({
                    'code': trade['code'],
                    'demand': trade['demand'],
                    'max_lanes': trade['max_lanes'],
                })

            input_data = {
                'problem_type': 'vessel_deployment_advanced',
                'vessel_sizes': vessel_sizes,
                'vessel_availability': vessel_availability,
                'trades': solver_trades,
                'v_min': v_min,
                'v_max': v_max,
            }

            results, error_msg, processing_time = (
                VesselDeploymentAdvancedSolver(input_data).solve()
                if solver_method == 'gurobi'
                else VesselDeploymentAdvancedOrtoolsSolver(input_data).solve()
            )
            context['processing_time_seconds'] = processing_time
            solver_label = 'Gurobi' if solver_method == 'gurobi' else 'OR-Tools SAT'

            if error_msg:
                context['error_message'] = error_msg
            elif results:
                context['results'] = results
                active_lanes = len(results['deployment_matrix'])
                context['success_message'] = (
                    f"[{solver_label}] 최적화 완료: 총 선박 {results['total_vessels_used']}척, "
                    f"활성 Lane {active_lanes}개 "
                    f"(소요시간: {processing_time}초)"
                )
            else:
                context['error_message'] = "최적화 결과를 가져오지 못했습니다."

        except ValueError as ve:
            context['error_message'] = f"입력값 오류: {ve}"
        except Exception as e:
            error_str = str(e)
            if 'size-limited license' in error_str or 'Model too large' in error_str:
                context['error_message'] = (
                    "Gurobi 제한 라이선스로는 이 규모의 모델을 풀 수 없습니다. "
                    "OR-Tools SAT 솔버를 사용하거나 Gurobi 정식 라이선스를 설치하세요."
                )
            else:
                context['error_message'] = f"처리 중 오류 발생: {e}"

    return render(request, 'complex_app/vessel_deployment_demo3.html', context)


def _parse_positive_float(source, key, label):
    value = float(source.get(key, 0))
    if value <= 0:
        raise ValueError(f"{label}는 0보다 커야 합니다.")
    return value


def _parse_non_negative_float(source, key, label):
    value = float(source.get(key, 0))
    if value < 0:
        raise ValueError(f"{label}는 0 이상이어야 합니다.")
    return value


def _parse_positive_int(source, key, label):
    value = int(source.get(key, 0))
    if value <= 0:
        raise ValueError(f"{label}는 1 이상이어야 합니다.")
    return value


@log_view_activity
def palletizing_demo_view(request):
    source = request.POST if request.method == 'POST' else request.GET
    submitted_num_types = int(source.get('num_types_to_show', source.get('num_types', 3)))
    submitted_num_types = max(1, min(5, submitted_num_types))
    solver_method = source.get('solver_method', 'heuristic')
    objective_function = source.get('objective_function', 'utilization')

    # 휴리스틱이면 목적함수는 적재율 최대화로 고정
    if solver_method == 'heuristic':
        objective_function = 'utilization'

    pallet_data = {
        'l': source.get('pallet_l', DEFAULT_PALLET['l']),
        'w': source.get('pallet_w', DEFAULT_PALLET['w']),
        'h': source.get('pallet_h', DEFAULT_PALLET['h']),
        'max_weight': source.get('pallet_max_weight', DEFAULT_PALLET['max_weight']),
    }

    box_types_data = []
    for idx in range(submitted_num_types):
        preset = DEFAULT_BOX_TYPES[idx]
        box_types_data.append({
            'id': source.get(f'box_{idx}_id', preset['id']),
            'l': source.get(f'box_{idx}_l', preset['l']),
            'w': source.get(f'box_{idx}_w', preset['w']),
            'h': source.get(f'box_{idx}_h', preset['h']),
            'weight': source.get(f'box_{idx}_weight', preset['weight']),
            'qty': source.get(f'box_{idx}_qty', preset['qty']),
            'rotatable': source.get(f'box_{idx}_rotatable', 'on' if preset['rotatable'] else ''),
        })

    context = {
        'active_model': 'Complex Optimization',
        'active_submenu': 'Palletizing Demo',
        'num_type_options': range(1, 6),
        'submitted_num_types': submitted_num_types,
        'solver_method': solver_method,
        'objective_function': objective_function,
        'pallet_data': pallet_data,
        'box_types_data': box_types_data,
        'results': None,
        'error_message': None,
        'success_message': None,
        'processing_time_seconds': "N/A",
    }

    if request.method == 'POST':
        try:
            parsed_pallet = {
                'l': _parse_positive_float(source, 'pallet_l', '팔렛 길이'),
                'w': _parse_positive_float(source, 'pallet_w', '팔렛 폭'),
                'h': _parse_positive_float(source, 'pallet_h', '팔렛 높이'),
                'max_weight': _parse_positive_float(source, 'pallet_max_weight', '최대 중량'),
            }

            parsed_box_types = []
            for idx in range(submitted_num_types):
                box_id = source.get(f'box_{idx}_id', f'BX{idx + 1}').strip() or f'BX{idx + 1}'
                parsed_box_types.append({
                    'id': box_id,
                    'l': _parse_positive_float(source, f'box_{idx}_l', f'박스 {idx + 1} 길이'),
                    'w': _parse_positive_float(source, f'box_{idx}_w', f'박스 {idx + 1} 폭'),
                    'h': _parse_positive_float(source, f'box_{idx}_h', f'박스 {idx + 1} 높이'),
                    'weight': _parse_non_negative_float(source, f'box_{idx}_weight', f'박스 {idx + 1} 중량'),
                    'qty': _parse_positive_int(source, f'box_{idx}_qty', f'박스 {idx + 1} 수량'),
                    'rotatable': source.get(f'box_{idx}_rotatable') == 'on',
                })

            input_data = {
                'problem_type': 'palletizing_3d',
                'objective_function': objective_function,
                'pallet': parsed_pallet,
                'box_types': parsed_box_types,
            }

            if solver_method == 'heuristic':
                results = PalletizingLogicSolver(input_data).solve()
                if not isinstance(results, dict) or 'summary' not in results:
                    context['error_message'] = "결과 데이터 형식 오류: 요약 정보를 찾을 수 없습니다."
                else:
                    context['results'] = results
                    summary = results['summary']
                    context['success_message'] = (
                        f"[휴리스틱] 적재 완료: {summary['placed_units']}/{summary['total_units']}개, "
                        f"적재율 {summary['utilization_percent']}%"
                    )
            else:
                results_data, error_msg_opt, processing_time = PalletizingSolver(input_data).solve()
                context['processing_time_seconds'] = processing_time
                if error_msg_opt:
                    context['error_message'] = error_msg_opt
                elif results_data:
                    context['results'] = results_data
                    summary = results_data['summary']
                    if objective_function == 'utilization':
                        context['success_message'] = (
                            f"[수리최적화] 적재 완료: 적재율 {summary['utilization_percent']}%, "
                            f"적재된 박스 {summary['placed_units']}/{summary['total_units']}개 "
                            f"(소요시간: {processing_time}초)"
                        )
                    elif objective_function == 'boxes':
                        context['success_message'] = (
                            f"[수리최적화] 적재 완료: 적재된 박스 {summary['placed_units']}/{summary['total_units']}개, "
                            f"적재율 {summary['utilization_percent']}% "
                            f"(소요시간: {processing_time}초)"
                        )
                    elif objective_function == 'weight':
                        context['success_message'] = (
                            f"[수리최적화] 적재 완료: 총 중량 {summary['total_weight']}kg, "
                            f"적재된 박스 {summary['placed_units']}/{summary['total_units']}개 "
                            f"(소요시간: {processing_time}초)"
                        )
                else:
                    context['error_message'] = "최적화 결과를 가져오지 못했습니다 (결과 없음)."

        except ValueError as ve:
            context['error_message'] = f"입력값 오류: {ve}"
        except Exception as e:
            context['error_message'] = f"처리 중 오류 발생: {e}"

    return render(request, 'complex_app/palletizing_demo.html', context)
