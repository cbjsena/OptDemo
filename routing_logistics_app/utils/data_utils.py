from common_utils.common_data_utils import save_json_data
import logging
import datetime  # 파일명 생성 등에 사용 가능

logger = logging.getLogger(__name__)

def create_vrp_json_data(form_data, submitted_num_customers, parsed_customer_locations):
    # --- 1. 입력 데이터 파싱 및 기본 유효성 검사 ---
    num_depots=1
    parsed_depot_location = {
        'x': float(form_data.get('depot_x', '0')),
        'y': float(form_data.get('depot_y', '0'))
    }
    for i in range(submitted_num_customers):
        parsed_customer_locations.append({
            'id': form_data.get(f'cust_{i}_id', f'C{i + 1}'),
            'x': float(form_data.get(f'cust_{i}_x')),  # 필수값으로 가정
            'y': float(form_data.get(f'cust_{i}_y'))  # 필수값으로 가정
        })

    num_vehicles_val_parsed = int(form_data.get('num_vehicles'))
    if num_vehicles_val_parsed <= 0:
        raise ValueError("차량 수는 1대 이상이어야 합니다.")

    logger.debug(f"Depot: {num_depots}, Customers: {len(parsed_customer_locations)}, Vehicles: {num_vehicles_val_parsed}")

    # --- 2. 입력 데이터 JSON 파일로 저장 ---
    input_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "problem_type": "VRP",
        "depot_location": parsed_depot_location,
        "customer_locations": parsed_customer_locations,
        "num_vehicles": num_vehicles_val_parsed,
        "num_depots":num_depots,
        # 추가적으로 저장하고 싶은 다른 form_data 항목들
        "form_parameters": {
            key: value for key, value in form_data.items() if key not in ['csrfmiddlewaretoken']
        }
    }
    return input_data


def save_vrp_json_data(input_data):
    num_depots = input_data.get('num_depots')
    num_vehicles = input_data.get('num_vehicles')
    num_customers = len(input_data.get('customer_locations'))
    filename_pattern = f"dep{num_depots}_cus{num_customers}_veh{num_vehicles}"
    return save_json_data(input_data, 'routing_vrp_data', filename_pattern)

