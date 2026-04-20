from django.shortcuts import render

from core.decorators import log_view_activity
from .solvers.palletizing_solver import PalletizingSolver


DEFAULT_PALLET = {
    'l': 120.0,
    'w': 100.0,
    'h': 150.0,
    'max_weight': 1200.0,
}

DEFAULT_BOX_TYPES = [
    {'id': 'BX1', 'l': 40.0, 'w': 30.0, 'h': 20.0, 'weight': 12.0, 'qty': 8, 'rotatable': True},
    {'id': 'BX2', 'l': 50.0, 'w': 40.0, 'h': 25.0, 'weight': 18.0, 'qty': 5, 'rotatable': True},
    {'id': 'BX3', 'l': 30.0, 'w': 20.0, 'h': 15.0, 'weight': 6.0, 'qty': 10, 'rotatable': False},
    {'id': 'BX4', 'l': 60.0, 'w': 40.0, 'h': 35.0, 'weight': 24.0, 'qty': 3, 'rotatable': True},
    {'id': 'BX5', 'l': 45.0, 'w': 35.0, 'h': 20.0, 'weight': 11.0, 'qty': 6, 'rotatable': True},
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
        'pallet_data': pallet_data,
        'box_types_data': box_types_data,
        'results': None,
        'error_message': None,
        'success_message': None,
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
                'pallet': parsed_pallet,
                'box_types': parsed_box_types,
            }
            context['results'] = PalletizingSolver(input_data).solve()
            summary = context['results']['summary']
            context['success_message'] = (
                f"적재 완료: {summary['placed_units']}/{summary['total_units']}개, "
                f"적재율 {summary['utilization_percent']}%"
            )
        except ValueError as ve:
            context['error_message'] = f"입력값 오류: {ve}"
        except Exception as e:
            context['error_message'] = f"처리 중 오류 발생: {e}"

    return render(request, 'complex_app/palletizing_demo.html', context)

