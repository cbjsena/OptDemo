from django.shortcuts import render

from core.decorators import log_view_activity
from .solvers.palletizing_solver import PalletizingLogicSolver, PalletizingSolver
from .solvers.vessel_deployment_solver import VesselDeploymentSolver


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

DEFAULT_ROUTES = [
    {'name': 'FE1', 'trade': 'FE', 'deployment': [15, 0, 0, 0, 0, 0, 0, 0]},
    {'name': 'FE2', 'trade': 'FE', 'deployment': [5, 3, 6, 0, 0, 0, 0, 0]},
    {'name': 'FE3', 'trade': 'FE', 'deployment': [0, 3, 7, 5, 0, 0, 0, 0]},
    {'name': 'FE4', 'trade': 'FE', 'deployment': [0, 0, 0, 10, 4, 0, 0, 0]},
    {'name': 'MD1', 'trade': 'MD', 'deployment': [0, 0, 0, 7, 5, 2, 0, 0]},
    {'name': 'MD2', 'trade': 'MD', 'deployment': [0, 0, 0, 0, 8, 5, 0, 0]},
    {'name': 'PS1', 'trade': 'PS', 'deployment': [0, 0, 0, 3, 3, 0, 0, 0]},
    {'name': 'PS2', 'trade': 'PS', 'deployment': [0, 0, 0, 0, 3, 3, 0, 0]},
    {'name': 'PS3', 'trade': 'PS', 'deployment': [0, 0, 0, 4, 2, 0, 0, 0]},
    {'name': 'PS4', 'trade': 'PS', 'deployment': [0, 0, 5, 1, 0, 0, 0, 0]},
    {'name': 'PS5', 'trade': 'PS', 'deployment': [0, 0, 0, 6, 1, 0, 0, 0]},
    {'name': 'PN1', 'trade': 'PN', 'deployment': [0, 0, 0, 0, 5, 1, 0, 0]},
    {'name': 'PN2', 'trade': 'PN', 'deployment': [0, 0, 0, 7, 0, 0, 0, 0]},
    {'name': 'EC1', 'trade': 'EC', 'deployment': [0, 0, 0, 10, 3, 0, 0, 0]},
    {'name': 'EC2', 'trade': 'EC', 'deployment': [0, 0, 0, 0, 8, 6, 0, 0]},
    {'name': 'EC3', 'trade': 'EC', 'deployment': [0, 0, 0, 5, 5, 2, 0, 0]},
    {'name': 'ME1', 'trade': 'ME', 'deployment': [0, 0, 0, 0, 0, 0, 4, 4]},
    {'name': 'ME2', 'trade': 'ME', 'deployment': [0, 0, 0, 0, 0, 0, 5, 4]},
]


def _compute_route_demand(deployment, vessel_sizes):
    """배치 수 × 선박 크기 합계로 수요 계산"""
    return sum(d * s for d, s in zip(deployment, vessel_sizes))


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
    """Vessel Deployment 데모 뷰"""
    source = request.POST if request.method == 'POST' else request.GET
    vessel_sizes = list(DEFAULT_VESSEL_SIZES)

    # 가용 수량 (POST에서 읽거나 기본값)
    vessel_availability = {}
    for size in vessel_sizes:
        key = f'avail_{size}'
        vessel_availability[str(size)] = int(source.get(key, DEFAULT_VESSEL_AVAILABILITY.get(str(size), 0)))

    # 항로 데이터 구성
    routes_data = []
    if request.method == 'POST':
        num_routes = int(source.get('num_routes', len(DEFAULT_ROUTES)))
        num_sizes = int(source.get('num_sizes', len(vessel_sizes)))
        # POST에서 vessel_sizes 복원
        vessel_sizes = []
        for s_idx in range(num_sizes):
            vessel_sizes.append(int(source.get(f'size_{s_idx}', DEFAULT_VESSEL_SIZES[s_idx])))

        for r_idx in range(num_routes):
            name = source.get(f'route_name_{r_idx}', f'R{r_idx+1}')
            trade = source.get(f'route_trade_{r_idx}', '')
            demand = int(source.get(f'demand_{r_idx}', 0))
            deployment = []
            for s_idx in range(len(vessel_sizes)):
                deployment.append(int(source.get(f'dep_{r_idx}_{s_idx}', 0)))
            routes_data.append({
                'name': name,
                'trade': trade,
                'demand': demand,
                'deployment': deployment,
                'total_vessels': sum(deployment),
            })
    else:
        for route in DEFAULT_ROUTES:
            demand = _compute_route_demand(route['deployment'], vessel_sizes)
            routes_data.append({
                'name': route['name'],
                'trade': route['trade'],
                'demand': demand,
                'deployment': list(route['deployment']),
                'total_vessels': sum(route['deployment']),
            })

    original_total_vessels = sum(r['total_vessels'] for r in routes_data)

    context = {
        'active_model': 'Complex Optimization',
        'active_submenu': 'vessel_deployment_demo',
        'vessel_sizes': vessel_sizes,
        'vessel_availability': vessel_availability,
        'routes_data': routes_data,
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
            solver_routes = []
            for route in routes_data:
                solver_routes.append({
                    'name': route['name'],
                    'trade': route['trade'],
                    'demand': route['demand'],
                })

            input_data = {
                'problem_type': 'vessel_deployment',
                'vessel_sizes': vessel_sizes,
                'vessel_availability': vessel_availability,
                'routes': solver_routes,
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
