import logging
import datetime  # 파일명 생성 등에 사용 가능
from common_utils.common_data_utils import save_json_data
logger = logging.getLogger(__name__)


def parse_allocation_budjet_data(form_data_from_post, num_items, total_budget):
    items_data = []
    for i in range(1, num_items + 1):
        name = form_data_from_post.get(f'item_{i}_name', f'item_{i}_name')
        return_coeff_str = form_data_from_post.get(f'item_{i}_return_coeff')
        min_alloc_str = form_data_from_post.get(f'item_{i}_min_alloc', '0')
        max_alloc_str = form_data_from_post.get(f'item_{i}_max_alloc', str(total_budget))
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
    json_data = create_allocation_budjet_json_data(total_budget, items_data)
    return items_data, json_data

def create_allocation_budjet_json_data(total_budget, items_data):
    generated_data = {
        "model_info": {
            "problem_type": "BudjetAllocationPlanning",
            "num_items_data": len(items_data)
        },
        "total_budget": total_budget,
        "items_data": items_data,
        "timestamp": datetime.datetime.now().isoformat(),
    }

    return generated_data


def save_allocation_budjet_json_data(json_data):
    num_item = len(json_data.get('items_data'))
    filename_pattern = f"item{num_item}"
    return save_json_data(json_data, 'allocation_budjet_data', filename_pattern)

def validate_data_center_data(global_constraints, server_types_data, service_demands_data):
    """
    데이터 센터 용량 계획 입력 데이터의 유효성을 검사합니다.
    오류가 있으면 오류 메시지 문자열을, 정상이면 None을 반환합니다.
    """
    # 1. 글로벌 제약 조건 유효성 검사
    required_global_keys = ['total_budget', 'total_power_kva', 'total_space_sqm']
    for key in required_global_keys:
        if key not in global_constraints:
            return f"글로벌 제약 조건에 필수 키 '{key}'가 없습니다."
        try:
            val = float(global_constraints[key])
            if val < 0:
                return f"글로벌 제약 조건 '{key}'의 값({val})은 음수가 될 수 없습니다."
            global_constraints[key] = val  # float으로 변환하여 업데이트
        except (ValueError, TypeError):
            return f"글로벌 제약 조건 '{key}'의 값('{global_constraints[key]}')이 올바른 숫자가 아닙니다."

    # 2. 서버 유형 데이터 유효성 검사
    if not server_types_data or not isinstance(server_types_data, list):
        return "서버 유형 데이터가 없거나 리스트 형식이 아닙니다."

    required_server_keys = ['id', 'cost', 'cpu_cores', 'ram_gb', 'storage_tb', 'power_kva', 'space_sqm']
    for i, server in enumerate(server_types_data):
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
            if not (server['cost'] >= 0 and server['cpu_cores'] >= 0 and server['ram_gb'] >= 0 and \
                    server['storage_tb'] >= 0 and server['power_kva'] >= 0 and server['space_sqm'] >= 0):
                return f"서버 유형 (ID: {server.get('id')})의 숫자 속성값은 음수가 될 수 없습니다."
        except (ValueError, TypeError) as e:
            return f"서버 유형 (ID: {server.get('id')})의 속성값 중 올바르지 않은 숫자 형식이 있습니다: {e}"

    # 3. 서비스 수요 데이터 유효성 검사
    if not service_demands_data or not isinstance(service_demands_data, list):
        return "서비스 수요 데이터가 없거나 리스트 형식이 아닙니다."

    required_service_keys = ['id', 'revenue_per_unit', 'req_cpu_cores', 'req_ram_gb', 'req_storage_tb', 'max_units']
    for i, service in enumerate(service_demands_data):
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

            if not (service['revenue_per_unit'] >= 0 and service['req_cpu_cores'] >= 0 and \
                    service['req_ram_gb'] >= 0 and service['req_storage_tb'] >= 0):
                return f"서비스 수요 (ID: {service.get('id')})의 숫자 속성값(수익, 요구자원)은 음수가 될 수 없습니다."
        except (ValueError, TypeError) as e:
            return f"서비스 수요 (ID: {service.get('id')})의 속성값 중 올바르지 않은 숫자 형식이 있습니다: {e}"

    logger.debug(f"Validated Global Constraints: {global_constraints}")
    logger.debug(f"Validated Server Types: {server_types_data}")
    logger.debug(f"Validated Service Demands: {service_demands_data}")
    return None  # 모든 유효성 검사 통과

def parse_allocation_data_center_data(form_data, submitted_num_server_types, submitted_num_services):
    parsed_global_constraints = {
        'total_budget': form_data.get('total_budget', '0'),  # 유효성 검사 함수에서 float 변환
        'total_power_kva': form_data.get('total_power_kva', '0'),
        'total_space_sqm': form_data.get('total_space_sqm', '0'),
    }
    parsed_server_types_data = []
    parsed_service_demands_data = []
    for i in range(submitted_num_server_types):
        parsed_server_types_data.append({
            'id': form_data.get(f'server_{i}_id', f'SrvType{i + 1}'),  # 기본 ID 제공
            'cost': form_data.get(f'server_{i}_cost', '0'),
            'cpu_cores': form_data.get(f'server_{i}_cpu_cores', '0'),
            'ram_gb': form_data.get(f'server_{i}_ram_gb', '0'),
            'storage_tb': form_data.get(f'server_{i}_storage_tb', '0'),
            'power_kva': form_data.get(f'server_{i}_power_kva', '0'),
            'space_sqm': form_data.get(f'server_{i}_space_sqm', '0'),
        })

    for i in range(submitted_num_services):
        parsed_service_demands_data.append({
            'id': form_data.get(f'service_{i}_id', f'Svc{i + 1}'),  # 기본 ID 제공
            'revenue_per_unit': form_data.get(f'service_{i}_revenue_per_unit', '0'),
            'req_cpu_cores': form_data.get(f'service_{i}_req_cpu_cores', '0'),
            'req_ram_gb': form_data.get(f'service_{i}_req_ram_gb', '0'),
            'req_storage_tb': form_data.get(f'service_{i}_req_storage_tb', '0'),
            'max_units': form_data.get(f'service_{i}_max_units'),  # None일 수 있음
        })
    return parsed_global_constraints, parsed_server_types_data, parsed_service_demands_data

def create_allocation_data_center_json_data(global_constraints, server_types_data, service_demands_data):
    json_data = {
        "model_info": {
            "problem_type": "DataCenterCapacityPlanning",
            "num_server_types": len(server_types_data),
            "num_services": len(service_demands_data)
        },
        "global_constraints": global_constraints,
        "server_types": server_types_data,
        "service_demands": service_demands_data,
        "timestamp": datetime.datetime.now().isoformat(),
    }

    return json_data

def save_allocation_data_center_json_data(json_data):
    num_server = len(json_data.get('server_types'))
    num_service = len(json_data.get('service_demands'))
    filename_pattern = f"svr{num_server}_svc{num_service}"
    return save_json_data(json_data, 'allocation_datacenter_data', filename_pattern)

def set_chart_data(results_data, parsed_global_constraints ):
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

