from common_utils.common_data_utils import save_json_data
import logging
import datetime  # 파일명 생성 등에 사용 가능
from common_utils.default_data import (
    preset_depot_location,
    preset_customer_locations,
    preset_num_customers,
    preset_num_vehicles,
    preset_vehicle_capacity
)
logger = logging.getLogger(__name__)

def create_vrp_json_data(form_data):
    # --- 1. 입력 데이터 파싱 및 기본 유효성 검사 ---
    num_depots=1
    parsed_depot_location = {
        'x': float(form_data.get('depot_x', '0')),
        'y': float(form_data.get('depot_y', '0'))
    }

    num_customers = int(form_data.get('num_customers', preset_num_customers))
    parsed_customer_locations=[]
    for i in range(num_customers):
        demand_val_str = form_data.get(f'cust_{i}_demand', '0')
        parsed_customer_locations.append({
            'id': form_data.get(f'cust_{i}_id', f'C{i + 1}'),
            'x': float(form_data.get(f'cust_{i}_x')),  # 필수값으로 가정
            'y': float(form_data.get(f'cust_{i}_y')),  # 필수값으로 가정
            'demand': int(demand_val_str) if demand_val_str.isdigit() else 0
        })

    num_vehicles = int(form_data.get('num_vehicles',preset_num_vehicles))
    if num_vehicles <= 0:
        raise ValueError("차량 수는 1대 이상이어야 합니다.")

    vehicle_capacity = int(form_data.get('vehicle_capacity', preset_vehicle_capacity))
    logger.debug(f"Depot: {parsed_depot_location}, Customers: {num_customers}, Vehicles: {num_vehicles}, Capacities: {vehicle_capacity}")

    # --- 2. 입력 데이터 JSON 파일로 저장 ---
    input_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "problem_type": form_data.get('problem_type'),
        "depot_location": parsed_depot_location,
        "customer_locations": parsed_customer_locations,
        "num_vehicles": num_vehicles,
        "num_depots":num_depots,
        "vehicle_capacity":vehicle_capacity,
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
    if "VRP" == input_data.get('problem_type'):
        dir ='routing_vrp_data'
    elif "CVRP" == input_data.get('problem_type'):
        dir ='routing_cvrp_data'
    return save_json_data(input_data, dir, filename_pattern)



