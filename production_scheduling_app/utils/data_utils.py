from common_utils.common_data_utils import save_json_data
import logging
import random
import datetime

logger = logging.getLogger(__name__)


def create_lot_sizing_json_data(form_data, num_periods):
    """
    폼 데이터로부터 Lot Sizing 문제 입력을 위한 딕셔너리를 생성하고 검증합니다.
    """
    logger.info("Creating and validating lot sizing input data from form.")
    demands = []
    setup_costs = []
    prod_costs = []
    holding_costs = []
    capacities = []
    for t in range(num_periods):
        try:
            demand = int(form_data.get(f'demand_{t}'))
            setup_cost = int(form_data.get(f'setup_cost_{t}'))
            prod_cost = int(form_data.get(f'prod_cost_{t}'))
            holding_cost = int(form_data.get(f'holding_cost_{t}'))
            capacity = form_data.get(f'capacity_{t}')

            if not all(isinstance(v, int) and v >= 0 for v in [demand, setup_cost, prod_cost, holding_cost]):
                raise ValueError(f"기간 {t + 1}의 입력값은 0 이상의 정수여야 합니다.")
            if capacity is None or not capacity.strip().isdigit():
                raise ValueError(f"기간 {t + 1}의 생산 능력(Capacity)이 올바른 숫자가 아닙니다.")

            demands.append(demand)
            setup_costs.append(setup_cost)
            prod_costs.append(prod_cost)
            holding_costs.append(holding_cost)
            capacities.append(int(capacity))

        except (ValueError, TypeError) as e:
            # 더 구체적인 오류 메시지를 위해 래핑
            raise ValueError(f"기간 {t + 1} 처리 중 오류 발생: {e}")

    input_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "problem_type": form_data.get('problem_type'),
        "num_periods": num_periods,
        'demands': demands,
        'setup_costs': setup_costs,
        'prod_costs': prod_costs,
        'holding_costs': holding_costs,
        'capacities': capacities
    }

    return input_data


def save_production_json_data(input_data):
    logger.info(f"Saving {input_data.get('problem_type')} data.")
    dir=''
    filename_pattern = ''
    if "lot_sizing" == input_data.get('problem_type'):
        num_periods = input_data.get('num_periods')
        dir = 'production_lot_sizing_data'
        filename_pattern = f"periods{num_periods}"
    elif "single_machine" == input_data.get('problem_type'):
        num_resources = input_data.get('num_resources')
        num_projects = input_data.get('num_projects')
        dir = 'production_single_machine_data'
        filename_pattern = f"resource{num_resources}_project{num_projects}"
    elif "flow_shop" == input_data.get('problem_type'):
        num_resources = input_data.get('num_resources')
        num_projects = input_data.get('num_projects')
        dir = 'production_flow_shop_data'
        filename_pattern = f"resource{num_resources}_project{num_projects}"
    elif "job_shop" == input_data.get('problem_type'):
        num_resources = input_data.get('num_resources')
        num_projects = input_data.get('num_projects')
        dir = 'production_job_shop_data'
        filename_pattern = f"resource{num_resources}_project{num_projects}"
    elif "rcpsp" == input_data.get('problem_type'):
        num_resources = input_data.get('num_resources')
        num_projects = input_data.get('num_projects')
        dir = 'production_rcpsp_data'
        filename_pattern = f"resource{num_resources}_project{num_projects}"

    return save_json_data(input_data, dir, filename_pattern)
