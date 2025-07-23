import logging
import datetime

from common_utils.common_data_utils import save_json_data
from core.decorators import log_data_creation

logger = logging.getLogger(__name__)

preset_total_budget = 1000
preset_budget_num_item = 3
preset_budget_items = [
    {'name': 'item_1', 'return_coefficient': '3.1', 'min_alloc': '0', 'max_alloc': '200'},
    {'name': 'item_2', 'return_coefficient': '2.1', 'min_alloc': '0', 'max_alloc': '300'},
    {'name': 'item_3', 'return_coefficient': '1.1', 'min_alloc': '0', 'max_alloc': '1000'},
    {'name': 'item_4', 'return_coefficient': '3.1', 'min_alloc': '0', 'max_alloc': '200'},
    {'name': 'item_5', 'return_coefficient': '2.1', 'min_alloc': '0', 'max_alloc': '300'},
    {'name': 'item_6', 'return_coefficient': '1.1', 'min_alloc': '0', 'max_alloc': '1000'},
    {'name': 'item_7', 'return_coefficient': '3.1', 'min_alloc': '0', 'max_alloc': '200'},
    {'name': 'item_8', 'return_coefficient': '2.1', 'min_alloc': '0', 'max_alloc': '300'},
    {'name': 'item_9', 'return_coefficient': '1.1', 'min_alloc': '0', 'max_alloc': '1000'},
    {'name': 'item_10', 'return_coefficient': '3.1', 'min_alloc': '0', 'max_alloc': '200'}
]
preset_datacenter_num_server_types = 2
preset_datacenter_num_services = 2
preset_datacenter_servers = [
    {'id': 'SrvA', 'cost': '500', 'cpu_cores': '48', 'ram_gb': '256', 'storage_tb': '10', 'power_kva': '0.5',
     'space_sqm': '0.2'},
    {'id': 'SrvB', 'cost': '300', 'cpu_cores': '32', 'ram_gb': '128', 'storage_tb': '5', 'power_kva': '0.3',
     'space_sqm': '0.1'},
    {'id': 'SrvC', 'cost': '800', 'cpu_cores': '128', 'ram_gb': '512', 'storage_tb': '20', 'power_kva': '0.8',
     'space_sqm': '0.3'}
]
preset_datacenter_services = [
    {'id': 'WebPool', 'revenue_per_unit': '100', 'req_cpu_cores': '4', 'req_ram_gb': '8',
     'req_storage_tb': '0.1', 'max_units': '50'},
    {'id': 'DBFarm', 'revenue_per_unit': '200', 'req_cpu_cores': '8', 'req_ram_gb': '16',
     'req_storage_tb': '0.5', 'max_units': '20'},
    {'id': 'BatchProc', 'revenue_per_unit': '150', 'req_cpu_cores': '16', 'req_ram_gb': '32',
     'req_storage_tb': '0.2', 'max_units': '30'}
]

preset_nurse_rostering_num_nurses = 15
preset_nurse_rostering_days = 7
preset_nurse_rostering_min_shifts = 2
preset_nurse_rostering_max_shifts = 7
preset_nurse_num_nurses_options= range(5, 21)
preset_nurse_num_days_options = [7, 10, 14, 21, 28]
preset_nurse_min_shifts_options = range(2, 14)
preset_nurse_max_shifts_options = range(7, 28)
preset_nurse_rostering_shifts = ['Day', 'Aft', 'Ngt']
preset_nurse_rostering_requests = {'Day': 4, 'Aft': 3, 'Ngt': 2}

preset_nurse_rostering_nurses_data = [
    {'id': 1, 'name': 'Nur1', 'skill': 'L'},
    {'id': 2, 'name': 'Nur2', 'skill': 'M'},
    {'id': 3, 'name': 'Nur3', 'skill': 'L'},
    {'id': 4, 'name': 'Nur4', 'skill': 'H'},
    {'id': 5, 'name': 'Nur5', 'skill': 'M'},
    {'id': 6, 'name': 'Nur6', 'skill': 'H'},
    {'id': 7, 'name': 'Nur7', 'skill': 'M'},
    {'id': 8, 'name': 'Nur8', 'skill': 'L'},
    {'id': 9, 'name': 'Nur9', 'skill': 'H'},
    {'id': 10, 'name': 'Nur10', 'skill': 'M'},
    {'id': 11, 'name': 'Nur11', 'skill': 'L'},
    {'id': 12, 'name': 'Nur12', 'skill': 'M'},
    {'id': 13, 'name': 'Nur13', 'skill': 'H'},
    {'id': 14, 'name': 'Nur14', 'skill': 'M'},
    {'id': 15, 'name': 'Nur15', 'skill': 'M'},
    {'id': 16, 'name': 'Nur16', 'skill': 'L'},
    {'id': 17, 'name': 'Nur17', 'skill': 'L'},
    {'id': 18, 'name': 'Nur18', 'skill': 'M'},
    {'id': 19, 'name': 'Nur19', 'skill': 'H'},
    {'id': 20, 'name': 'Nur2O', 'skill': 'M'},
    # {'id': i, 'name': f'Nur{chr(65+i)}', 'skill': random.choice(['H', 'M', 'L'])} for i in range(15)
]
preset_nurse_rostering_skill_options = ['H', 'M', 'L']
preset_nurse_rostering_shift_requirements = {
    'Day': {'H': '1', 'M': '2', 'L': '1'},
    'Aft': {'H': '1', 'M': '2', 'L': '1'},
    'Ngt': {'H': '1', 'L': '1'}
}
preset_nurse_rostering_enabled_fairness = ['fair_weekends', 'fair_nights', 'fair_offs']  # 'fair_weekends'


@log_data_creation
def create_budget_allocation_json_data(form_data):
    total_budget_str = form_data.get('total_budget')
    num_items = int(form_data.get('num_items'))
    if not total_budget_str:
        raise ValueError("총 예산이 입력되지 않았습니다.")
    total_budget = int(total_budget_str)
    if total_budget < 0:
        raise ValueError("총 예산은 음수가 될 수 없습니다.")

    items_data = []
    for i in range(num_items):
        name = form_data.get(f'item_{i}_name')
        return_coeff_str = form_data.get(f'item_{i}_return_coefficient')
        min_alloc_str = form_data.get(f'item_{i}_min_alloc', '0')
        max_alloc_str = form_data.get(f'item_{i}_max_alloc', str(total_budget))
        if not return_coeff_str:
            raise ValueError(f"'{name}'의 기대 수익률 계수가 입력되지 않았습니다.")

        try:
            return_coeff = float(return_coeff_str)
            min_alloc = float(min_alloc_str)
            max_alloc = float(max_alloc_str)
        except ValueError:
            raise ValueError(f"'{name}'의 숫자 입력값(수익률, 최소/최대 투자액)이 올바르지 않습니다.")

        if min_alloc < 0 or max_alloc < 0:
            raise ValueError(f"'{name}'의 최소/최대 투자액은 음수가 될 수 없습니다.")
        if min_alloc > max_alloc:
            raise ValueError(f"'{name}'의 최소 투자액({min_alloc})이 최대 투자액({max_alloc})보다 클 수 없습니다.")

        items_data.append({
            'name': name,
            'return_coefficient': return_coeff,
            'min_alloc': min_alloc,
            'max_alloc': max_alloc
        })

    input_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "problem_type": form_data.get('problem_type'),
        "total_budget": total_budget,
        "num_items": num_items,
        "items_data": items_data
    }
    return input_data


@log_data_creation
def create_datacenter_allocation_json_data(form_data):
    global_constraints = {
        'total_budget': form_data.get('total_budget', '0'),  # 유효성 검사 함수에서 float 변환
        'total_power_kva': form_data.get('total_power_kva', '0'),
        'total_space_sqm': form_data.get('total_space_sqm', '0'),
    }

    num_server_types = int(form_data.get('num_server_types', '0'))
    num_services = int(form_data.get('num_services', '0'))

    # 1. 글로벌 제약 조건 유효성 검사
    required_global_keys = ['total_budget', 'total_power_kva', 'total_space_sqm']
    for key in required_global_keys:
        if key not in global_constraints:
            return f"글로벌 제약 조건에 필수 키 '{key}'가 없습니다."
        try:
            val = int(global_constraints[key])
            if val < 0:
                return f"글로벌 제약 조건 '{key}'의 값({val})은 음수가 될 수 없습니다."
            global_constraints[key] = val
        except (ValueError, TypeError):
            return f"글로벌 제약 조건 '{key}'의 값('{global_constraints[key]}')이 올바른 숫자가 아닙니다."

    server_data = []
    for i in range(num_server_types):
        server_data.append({
            'id': form_data.get(f'server_{i}_id', f'SrvType{i + 1}'),  # 기본 ID 제공
            'cost': form_data.get(f'server_{i}_cost', '0'),
            'cpu_cores': form_data.get(f'server_{i}_cpu_cores', '0'),
            'ram_gb': form_data.get(f'server_{i}_ram_gb', '0'),
            'storage_tb': form_data.get(f'server_{i}_storage_tb', '0'),
            'power_kva': form_data.get(f'server_{i}_power_kva', '0'),
            'space_sqm': form_data.get(f'server_{i}_space_sqm', '0'),
        })
    # 2. 서버 유형 데이터 유효성 검사
    if not server_data or not isinstance(server_data, list):
        return "서버 유형 데이터가 없거나 리스트 형식이 아닙니다."

    required_server_keys = ['id', 'cost', 'cpu_cores', 'ram_gb', 'storage_tb', 'power_kva', 'space_sqm']
    for i, server in enumerate(server_data):
        if not isinstance(server, dict):
            return f"서버 유형 데이터 (인덱스 {i})가 딕셔너리 형식이 아닙니다."
        for key in required_server_keys:
            if key not in server:
                return f"서버 유형 (ID: {server.get('id', f'인덱스 {i}')})에 필수 키 '{key}'가 없습니다."
        try:
            server['cost'] = float(server['cost'])
            server['cpu_cores'] = int(server['cpu_cores'])
            server['ram_gb'] = int(server['ram_gb'])
            server['storage_tb'] = float(server['storage_tb'])
            server['power_kva'] = float(server['power_kva'])
            server['space_sqm'] = float(server['space_sqm'])
            if not (server['cost'] >= 0 and server['cpu_cores'] >= 0 and server['ram_gb'] >= 0 and
                    server['storage_tb'] >= 0 and server['power_kva'] >= 0 and server['space_sqm'] >= 0):
                return f"서버 유형 (ID: {server.get('id')})의 숫자 속성값은 음수가 될 수 없습니다."
        except (ValueError, TypeError) as e:
            return f"서버 유형 (ID: {server.get('id')})의 속성값 중 올바르지 않은 숫자 형식이 있습니다: {e}"

    demand_data = []
    for i in range(num_services):
        demand_data.append({
            'id': form_data.get(f'service_{i}_id', f'Svc{i + 1}'),  # 기본 ID 제공
            'revenue_per_unit': form_data.get(f'service_{i}_revenue_per_unit', '0'),
            'req_cpu_cores': form_data.get(f'service_{i}_req_cpu_cores', '0'),
            'req_ram_gb': form_data.get(f'service_{i}_req_ram_gb', '0'),
            'req_storage_tb': form_data.get(f'service_{i}_req_storage_tb', '0'),
            'max_units': form_data.get(f'service_{i}_max_units'),  # None일 수 있음
        })

    # 3. 서비스 수요 데이터 유효성 검사
    if not demand_data or not isinstance(demand_data, list):
        return "서비스 수요 데이터가 없거나 리스트 형식이 아닙니다."

    required_service_keys = ['id', 'revenue_per_unit', 'req_cpu_cores', 'req_ram_gb', 'req_storage_tb', 'max_units']
    for i, service in enumerate(demand_data):
        if not isinstance(service, dict):
            return f"서비스 수요 데이터 (인덱스 {i})가 딕셔너리 형식이 아닙니다."
        for key in required_service_keys:
            if key not in service and key != 'max_units':  # max_units는 None일 수 있음
                return f"서비스 수요 (ID: {service.get('id', f'인덱스 {i}')})에 필수 키 '{key}'가 없습니다."
        try:
            service['revenue_per_unit'] = float(service['revenue_per_unit'])
            service['req_cpu_cores'] = int(service['req_cpu_cores'])
            service['req_ram_gb'] = int(service['req_ram_gb'])
            service['req_storage_tb'] = float(service['req_storage_tb'])
            if service.get('max_units') is not None:  # None이 아닐 때만 정수 변환 시도
                service['max_units'] = int(service['max_units'])
                if service['max_units'] < 0: return f"서비스 수요 (ID: {service.get('id')})의 최대 유닛 수는 음수가 될 수 없습니다."

            if not (service['revenue_per_unit'] >= 0 and service['req_cpu_cores'] >= 0 and
                    service['req_ram_gb'] >= 0 and service['req_storage_tb'] >= 0):
                return f"서비스 수요 (ID: {service.get('id')})의 숫자 속성값(수익, 요구자원)은 음수가 될 수 없습니다."
        except (ValueError, TypeError) as e:
            return f"서비스 수요 (ID: {service.get('id')})의 속성값 중 올바르지 않은 숫자 형식이 있습니다: {e}"

    input_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "problem_type": form_data.get('problem_type'),
        "num_server_types": len(server_data),
        "num_services": len(demand_data),
        "global_constraints": global_constraints,
        "server_data": server_data,
        "demand_data": demand_data
    }

    return input_data


def set_datacenter_chart_data(results_data, parsed_global_constraints):
    chart_data = {}
    # 1. 구매 서버 수량 차트 데이터
    if results_data.get('purchased_servers'):
        chart_data['purchased_servers'] = {
            'labels': [s['type_id'] for s in results_data['purchased_servers']],
            'data': [s['count'] for s in results_data['purchased_servers']],
        }

    # 2. 자원 활용률 차트 데이터
    # 예산 활용률
    total_budget_input = parsed_global_constraints.get('total_budget', 0)
    used_budget = results_data.get('total_server_cost', 0)
    chart_data['budget_utilization'] = {
        'used': used_budget,
        'available': total_budget_input,
        'percent': round((used_budget / total_budget_input) * 100, 1) if total_budget_input > 0 else 0
    }
    # 전력 활용률
    total_power_input = parsed_global_constraints.get('total_power_kva', 0)
    used_power = results_data.get('total_power_used', 0)
    chart_data['power_utilization'] = {
        'used': used_power,
        'available': total_power_input,
        'percent': round((used_power / total_power_input) * 100, 1) if total_power_input > 0 else 0
    }
    # 공간 활용률
    total_space_input = parsed_global_constraints.get('total_space_sqm', 0)
    used_space = results_data.get('total_space_used', 0)
    chart_data['space_utilization'] = {
        'used': used_space,
        'available': total_space_input,
        'percent': round((used_space / total_space_input) * 100, 1) if total_space_input > 0 else 0
    }
    return chart_data


@log_data_creation
def create_nurse_rostering_json_data(form_data):
    num_nurses = int(form_data.get('num_nurses'))
    num_days = int(form_data.get('num_days'))
    min_shifts = int(form_data.get('min_shifts'))
    max_shifts = int(form_data.get('max_shifts'))

    day_shift_requests = []
    for d in range(preset_nurse_rostering_days):
        days = []
        for s_idx, s_name in enumerate(preset_nurse_rostering_shifts):
            required = int(form_data.get(f'shift_{s_idx}_req'))
            days.append(required)
        day_shift_requests.append(days)

    schedule_weekdays = get_schedule_weekdays(num_days)
    weekend_days = [i for i, day_name in enumerate(schedule_weekdays) if day_name in ['토', '일']]

    input_data = {
        "problem_type": form_data.get('problem_type'),
        'num_nurses': num_nurses,
        'num_days': num_days,
        'shifts': preset_nurse_rostering_shifts,
        'day_shift_requests': day_shift_requests,
        'min_shifts_per_nurse': min_shifts,
        'max_shifts_per_nurse': max_shifts,
        'schedule_weekdays': schedule_weekdays,
        'weekend_days': weekend_days
    }

    return input_data


@log_data_creation
def create_nurse_rostering_advanced_json_data(form_data):
    num_nurses = int(form_data.get('num_nurses'))
    num_days = int(form_data.get('num_days'))
    shifts = preset_nurse_rostering_shifts

    # 요일 및 주말 정보 생성
    today = datetime.date.today()
    schedule_dates = [today + datetime.timedelta(days=i) for i in range(num_days)]
    weekdays = ["월", "화", "수", "목", "금", "토", "일"]
    schedule_weekdays = [weekdays[d.weekday()] for d in schedule_dates]
    # weekend_days = [i for i, day_name in enumerate(schedule_weekdays) if day_name in ['토', '일']]

    # 간호사 정보 파싱
    nurses_data = [
        {'id': i, 'name': form_data.get(f'nurse_{i}_name'), 'skill': form_data.get(f'nurse_{i}_skill')}
        for i in range(num_nurses)
    ]
    # 시프트별 필요인원 파싱
    skill_reqs = {
        s_name: {
            skill: int(form_data.get(f'req_{s_name}_{skill}', 0))
            for skill in preset_nurse_rostering_skill_options
        } for s_name in shifts
    }

    vacation_reqs = {
        i: [int(d.strip()) - 1 for d in form_data.get(f'nurse_{i}_vacation', '').split(',') if d.strip().isdigit()] for
        i in range(num_nurses)}
    weekend_days = [i for i, day_name in enumerate(schedule_weekdays) if day_name in ['토', '일']]

    input_data = {
        "problem_type": form_data.get('problem_type'),
        'nurses_data': nurses_data,
        'num_nurses': num_nurses,
        'num_days': num_days,
        'shifts': shifts,
        'skill_requirements': skill_reqs,
        'vacation_requests': vacation_reqs,
        'enabled_fairness': form_data.getlist('fairness_options'),
        'weekend_days': weekend_days,
        'min_shifts_per_nurse': int(form_data.get('min_shifts', 5)),
        'max_shifts_per_nurse': int(form_data.get('max_shifts', 8)),
    }

    return input_data


def save_allocation_json_data(input_data):
    problem_type = input_data.get('problem_type')
    model_data_type = f'allocation_{problem_type}_data'
    filename_pattern = ''
    if "budget" == problem_type:
        num_items = input_data.get('num_items')
        filename_pattern = f"item{num_items}"
    elif "datacenter" == problem_type:
        num_server_types = input_data.get('num_server_types')
        num_services = input_data.get('num_services')
        filename_pattern = f"svr{num_server_types}_svc{num_services}"
    elif "nurse_rostering" == problem_type:
        num_nurses = input_data.get('num_nurses')
        num_days = input_data.get('num_days')
        filename_pattern = f"nurse{num_nurses}_day{num_days}"

    return save_json_data(input_data, model_data_type, filename_pattern)


def get_schedule_weekdays(num_days):
    today = datetime.date.today()
    schedule_dates = [today + datetime.timedelta(days=i) for i in range(num_days)]
    weekdays = ["월", "화", "수", "목", "금", "토", "일"]
    schedule_weekdays = [weekdays[d.weekday()] for d in schedule_dates]

    return schedule_weekdays
