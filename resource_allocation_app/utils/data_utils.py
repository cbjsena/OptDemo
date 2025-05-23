from django.conf import settings

import os
import json
import logging
import datetime  # 파일명 생성 등에 사용 가능

logger = logging.getLogger(__name__)


def parse_data_center_data(form_data, submitted_num_server_types, submitted_num_services):
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
    generated_data = {
        "model_info": {
            "problem_type": "DataCenterCapacityPlanning",
            "num_server_types": 2,
            "num_services": 2
        },
        "global_constraints": global_constraints,
        "server_types": server_types_data,
        "service_demands": service_demands_data,
        "timestamp": datetime.datetime.now().isoformat(),
    }

    return generated_data


def save_allocation_data_center_json_data(global_constraints, server_types_data, service_demands_data,
                                          directory_setting_name):
    """
    입력 데이터를 JSON 파일로 저장합니다.
    성공 시 저장된 파일명을, 실패 시 None을 반환합니다.
    """
    filename_prefix = 'datacenter_input'
    generated_data=create_allocation_data_center_json_data(global_constraints, server_types_data, service_demands_data)
    data_dir_path_str = getattr(settings, directory_setting_name, None)
    if not data_dir_path_str:
        logger.warning(f"{directory_setting_name} not configured in settings. Input data will not be saved.")
        return None, "서버 저장 경로가 설정되지 않아 입력 데이터를 저장할 수 없습니다."

    try:
        data_dir = str(data_dir_path_str)
        os.makedirs(data_dir, exist_ok=True)

        num_server = len(generated_data.get('server_types'))
        num_service = len(generated_data.get('service_demands'))
        timestamp_str = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        filename = f"{filename_prefix}_svr{num_server}_svc{num_service}_{timestamp_str}.json"
        filepath = os.path.join(data_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(generated_data, f, indent=4, ensure_ascii=False)
        logger.info(f"Input data saved to: {filepath}")
        return filename, None # 성공 시 파일명과 None (오류 없음) 반환
    except IOError as e:
        logger.error(f"Failed to save input data to {filepath}: {e}", exc_info=True)
        return None, f"입력 데이터를 파일로 저장하는 데 실패했습니다: {e}"
    except Exception as e:
        logger.error(f"Unexpected error during data saving: {e}", exc_info=True)
        return None, f"입력 데이터 저장 중 예상치 못한 오류 발생: {e}"