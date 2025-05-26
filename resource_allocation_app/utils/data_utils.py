from django.conf import settings

import os
import json
import logging
import datetime  # 파일명 생성 등에 사용 가능

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
    timestamp_str = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    filename = f"item{num_item}_{timestamp_str}.json"
    return save_json_data(json_data, 'allocation_budjet_input', filename)

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

def save_allocation_data_center_json_data(global_constraints, server_types_data, service_demands_data):
    generated_data=create_allocation_data_center_json_data(global_constraints, server_types_data, service_demands_data)
    num_server = len(generated_data.get('server_types'))
    num_service = len(generated_data.get('service_demands'))
    timestamp_str = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    filename = f"svr{num_server}_svc{num_service}_{timestamp_str}.json"
    return save_json_data(generated_data, 'allocation_datacenter_input', filename)


def save_json_data(generated_data, model_data_type, file_name):
    """
    입력 데이터를 JSON 파일로 저장합니다.
    성공 시 저장된 파일명을, 실패 시 None을 반환합니다.
    """
    data_dir_path_str = settings.DEMO_DIR_MAP[model_data_type]
    if not data_dir_path_str:
        logger.warning(f"{data_dir_path_str} not configured in settings. Input data will not be saved.")
        return None, "서버 저장 경로가 설정되지 않아 입력 데이터를 저장할 수 없습니다."

    try:
        data_dir = str(data_dir_path_str)
        os.makedirs(data_dir, exist_ok=True)
        filepath = os.path.join(data_dir, file_name)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(generated_data, f, indent=4, ensure_ascii=False)
        logger.info(f"Input data saved to: {filepath}")
        return file_name, None # 성공 시 파일명과 None (오류 없음) 반환
    except IOError as e:
        logger.error(f"Failed to save input data to {file_name}: {e}", exc_info=True)
        return None, f"입력 데이터를 파일로 저장하는 데 실패했습니다: {e}"
    except Exception as e:
        logger.error(f"Unexpected error during data saving: {e}", exc_info=True)
        return None, f"입력 데이터 저장 중 예상치 못한 오류 발생: {e}"