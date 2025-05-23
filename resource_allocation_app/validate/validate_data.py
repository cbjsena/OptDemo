import logging

logger = logging.getLogger(__name__)

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