from django.shortcuts import render
from common_utils.run_routing_opt import *
from .utils.data_utils import *
import random
import json
import logging

logger = logging.getLogger(__name__)

def vrp_introduction_view(request):
    """
    Vehicle Routing Problem (VRP) introduction page.
    """
    context = {
        'active_model': 'Routing & Logistics', # 대메뉴 활성화용
        'active_submenu_category': 'vehicle_routing_problems', # 사이드바 내 VRP 관련 그룹 활성화용
        'active_submenu': 'vrp_introduction' # 현재 페이지 활성화용
    }
    logger.debug("Rendering VRP introduction page.")
    return render(request, 'routing_logistics_app/vrp_introduction.html', context)


def vrp_advanced(request):
    context = {
        'active_model': 'Routing & Logistics',  # 대메뉴 활성화용
        'active_submenu_category': 'vehicle_routing_problems',  # 사이드바 내 VRP 관련 그룹 활성화용
        'active_submenu': 'vrp_advanced'  # 현재 페이지 활성화용
    }
    logger.debug("Rendering VRP introduction page.")
    return render(request, 'routing_logistics_app/vrp_advanced.html', context)

def vrp_demo_view(request):
    """
    Vehicle Routing Problem (VRP) introduction page.
    """
    default_num_depots=1
    default_num_customers = 3
    default_num_vehicles = 1
    form_data = {}

    if request.method == 'GET':
        submitted_num_customers = int(request.GET.get('num_customers_to_show', default_num_customers))
        submitted_num_vehicles = int(request.GET.get('num_vehicles_to_show', default_num_vehicles))
        submitted_num_customers = max(1, min(10, submitted_num_customers))
        submitted_num_vehicles = max(1, min(5, submitted_num_vehicles))

        form_data['depot_x'] = request.GET.get('depot_x', '0')
        form_data['depot_y'] = request.GET.get('depot_y', '0')
        for i in range(submitted_num_customers):
            form_data[f'cust_{i}_id'] = request.GET.get(f'cust_{i}_id', f'C{i + 1}')
            form_data[f'cust_{i}_x'] = request.GET.get(f'cust_{i}_x', str(random.randint(-10, 10)))
            form_data[f'cust_{i}_y'] = request.GET.get(f'cust_{i}_y', str(random.randint(-10, 10)))
    elif request.method == 'POST':
        form_data = request.POST.copy()
        submitted_num_customers = int(form_data.get('num_customers', default_num_customers))
        submitted_num_vehicles = int(form_data.get('num_vehicles', default_num_vehicles))

    context = {
        'active_model': 'Routing & Logistics',
        'active_submenu_category': 'vehicle_routing_problems',
        'active_submenu_case': 'vrp_demo',
        'form_data': form_data,
        'vrp_results': None, 'error_message': None, 'success_message': None, 'info_message': None,
        'processing_time_seconds': "N/A",
        'num_customers_options': range(1, 11),
        'num_vehicles_options': range(1, 6),
        'submitted_num_customers': submitted_num_customers,
        'submitted_num_vehicles': submitted_num_vehicles,
        'plot_data': None
    }

    if request.method == 'POST':
        logger.info("VRP Demo POST request processing.")
        parsed_customer_locations = []
        logger.info("VRP Demo POST request received.")

        try:
            input_data = create_vrp_json_data(form_data,submitted_num_customers, parsed_customer_locations )
            saved_filename, save_error = save_vrp_json_data(input_data)
            if save_error:
                context['error_message'] = (context.get('error_message', '') + " " + save_error).strip()  # 기존 에러에 추가
            elif saved_filename:
                context['success_save_message'] = f" 입력 데이터가 '{saved_filename}'으로 서버에 저장.".strip()

            vrp_results_data, error_msg_opt, processing_time_ms = run_vrp_optimizer(input_data)
            context[
                'processing_time_seconds'] = f"{(processing_time_ms / 1000.0):.3f}" if processing_time_ms is not None else "N/A"

            if error_msg_opt:
                context['error_message'] = error_msg_opt
            elif vrp_results_data and vrp_results_data['routes']:
                context['vrp_results'] = vrp_results_data
                context['success_message'] = f"VRP 최적 경로 계산 완료! 총 거리: {vrp_results_data['total_distance']:.2f}"
                logger.info(f"VRP optimization successful. Total distance: {vrp_results_data['total_distance']:.2f}")

                # --- 차트용 데이터 준비 ---
                plot_data = {'locations': [], 'routes': [], 'depot_index': 0}
                # 모든 위치 (차고지 + 고객)
                depot_location = input_data.get('depot_location')
                customer_locations = input_data.get('customer_locations')
                plot_data['locations'].append({'id': 'Depot', 'x': depot_location['x'], 'y': depot_location['y']})
                for i, cust_loc in enumerate(customer_locations):
                    plot_data['locations'].append({'id': cust_loc['id'], 'x': cust_loc['x'], 'y': cust_loc['y']})

                # 경로 데이터 (좌표 시퀀스)
                for route_info in vrp_results_data['routes']:
                    plot_data['routes'].append({
                        'vehicle_id': route_info['vehicle_id'],
                        'path_coords': route_info['route_locations']  # 이미 (x,y) 튜플의 리스트임
                    })
                context['plot_data'] = json.dumps(plot_data)  # JSON 문자열로 전달
                logger.debug(f"Plot data prepared: {plot_data}")


            else:  # 결과도 없고 명시적 에러도 없는 경우 (예: 해를 못 찾았지만 오류는 아닌 상태)
                context['error_message'] = "최적 경로를 찾지 못했거나 방문할 고객이 없습니다."
                if vrp_results_data and not vrp_results_data['routes']:
                    context['info_message'] = "모든 차량이 유휴 상태입니다 (방문할 고객이 없거나 할당 불가)."


        except ValueError as ve:
            context['error_message'] = f"입력값 오류: {str(ve)}"
            logger.error(f"ValueError in vrp_demo_view (POST): {ve}", exc_info=True)
        except Exception as e:
            context['error_message'] = f"처리 중 오류 발생: {str(e)}"
            logger.error(f"Unexpected error in vrp_demo_view (POST): {e}", exc_info=True)

    return render(request, 'routing_logistics_app/vrp_demo.html', context)


def cvrp_introduction_view(request):
    context = {
        'active_model': 'Routing & Logistics',  # 대메뉴 활성화용
        'active_submenu_category': 'capacitated_vehicle_routing_problems',  # 사이드바 내 VRP 관련 그룹 활성화용
        'active_submenu': 'cvrp_introduction'  # 현재 페이지 활성화용
    }
    logger.debug("Rendering CVRP introduction page.")
    return render(request, 'routing_logistics_app/cvrp_introduction.html', context)


def cvrp_demo_view(request):
    context = {
        'active_model': 'Routing & Logistics',  # 대메뉴 활성화용
        'active_submenu_category': 'capacitated_vehicle_routing_problems',  # 사이드바 내 VRP 관련 그룹 활성화용
        'active_submenu': 'cvrp_demo'  # 현재 페이지 활성화용
    }
    logger.debug("Rendering CVRP demo page.")
    return render(request, 'routing_logistics_app/cvrp_demo.html', context)


def pdp_introduction_view(request):
    context = {
        'active_model': 'Routing & Logistics',  # 대메뉴 활성화용
        'active_submenu_category': 'pickup_delivery_problem',  # 사이드바 내 VRP 관련 그룹 활성화용
        'active_submenu': 'pdp_introduction'  # 현재 페이지 활성화용
    }
    logger.debug("Rendering PDP introduction page.")
    return render(request, 'routing_logistics_app/pdp_introduction.html', context)


def pdp_demo_view(request):
    context = {
        'active_model': 'Routing & Logistics',  # 대메뉴 활성화용
        'active_submenu_category': 'pickup_delivery_problem',  # 사이드바 내 VRP 관련 그룹 활성화용
        'active_submenu': 'pdp_demo'  # 현재 페이지 활성화용
    }
    logger.debug("Rendering PDP demo page.")
    return render(request, 'routing_logistics_app/pdp_demo.html', context)


