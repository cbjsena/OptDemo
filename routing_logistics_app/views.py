from django.conf import settings
from django.shortcuts import render

from common_utils.run_routing_opt import *
from common_utils.data_utils_routing import *

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
    form_data = {}
    model_name = 'VRP'
    if request.method == 'GET':
        logger.info(f'{model_name} Demo GET request processing.')
        submitted_num_customers = int(request.GET.get('num_customers_to_show', preset_num_customers))
        submitted_num_vehicles = int(request.GET.get('num_vehicles_to_show', preset_num_vehicles))
        submitted_vehicle_capa = int(request.GET.get('num_vehicle_capa_to_show', preset_vehicle_capacity))
        submitted_num_customers = max(1, min(10, submitted_num_customers))
        submitted_num_vehicles = max(1, min(5, submitted_num_vehicles))
        submitted_vehicle_capa = max(50, min(100, submitted_vehicle_capa))

        form_data['depot_x'] = preset_depot_location.get('x')
        form_data['depot_y'] = preset_depot_location.get('y')
        for i in range(len(preset_customer_locations)):
            preset = preset_customer_locations[i % len(preset_customer_locations)]
            for key, default_val in preset.items():
                form_data[f'cust_{i}_{key}'] = request.GET.get(f'cust_{i}_{key}', default_val)

    elif request.method == 'POST':
        form_data = request.POST.copy()
        submitted_num_customers = int(form_data.get('num_customers', preset_num_customers))
        submitted_num_vehicles = int(form_data.get('num_vehicles', preset_num_vehicles))
        submitted_vehicle_capa= int(form_data.get('vehicle_capa', preset_vehicle_capacity))

    context = {
        'active_model': 'Routing & Logistics',
        'active_submenu_category': 'vehicle_routing_problems',
        'active_submenu': 'VRP Demo',
        'form_data': form_data,
        'opt_results': None,
        'error_message': None, 'success_message': None,
        'processing_time_seconds': "N/A",
        'num_customers_options': range(1, 11),
        'num_vehicles_options': range(1, 6),
        'vehicle_capa_options': range(50,210,10),
        'submitted_num_customers': submitted_num_customers,
        'submitted_num_vehicles': submitted_num_vehicles,
        'submitted_vehicle_capa' : submitted_vehicle_capa,
        'plot_data': None
    }

    if request.method == 'POST':
        logger.info(f'{model_name} Demo POST request processing.')

        try:
            # 1. 데이터 파일 새성 및 검증
            input_data = create_vrp_json_data(form_data)

            # 2. 파일 저장
            if settings.SAVE_DATA_FILE:
                success_save_message, save_error = save_vrp_json_data(input_data)
                if save_error:
                    context['error_message'] = (context.get('error_message', '') + " " + save_error).strip()  # 기존 에러에 추가
                elif success_save_message:
                    context['success_save_message'] = success_save_message

            # 3. 최적화 실행
            results_data, error_msg_opt, processing_time = run_vrp_optimizer(input_data)
            context['processing_time_seconds'] = processing_time

            if error_msg_opt:
                context['error_message'] = error_msg_opt
            elif results_data and results_data['routes']:
                context['opt_results'] = results_data
                success_message = f"{model_name} 최적 경로 계산 완료! 총 거리: {results_data['total_distance']:.2f}"
                context['success_message'] = success_message
                logger.info(success_message)

                # 차트용 데이터 준비
                plot_data = {'locations': [], 'routes': [], 'depot_index': 0}
                # 모든 위치 (차고지 + 고객)
                depot_location = input_data.get('depot_location')
                customer_locations = input_data.get('customer_locations')
                plot_data['locations'].append({'id': 'Depot', 'x': depot_location['x'], 'y': depot_location['y']})
                for i, cust_loc in enumerate(customer_locations):
                    plot_data['locations'].append(
                        {'id': cust_loc['id'], 'x': cust_loc['x'], 'y': cust_loc['y']})

                # 경로 데이터 (좌표 시퀀스)
                for route_info in results_data['routes']:
                    plot_data['routes'].append({
                        'vehicle_id': route_info['vehicle_id'],
                        'path_coords': route_info['route_locations']  # 이미 (x,y) 튜플의 리스트임
                    })
                context['plot_data'] = json.dumps(plot_data)  # JSON 문자열로 전달
                logger.debug(f"Plot data prepared: {plot_data}")
            else:  # 결과도 없고 명시적 에러도 없는 경우 (예: 해를 못 찾았지만 오류는 아닌 상태)
                current_error = context.get('error_message', '')
                default_opt_error = "최적 경로를 찾지 못했거나 방문할 고객이 없습니다."
                context['error_message'] = (
                        current_error + " " + default_opt_error).strip() if current_error else default_opt_error
        except ValueError as ve:
            context['error_message'] = f"입력값 오류: {str(ve)}"
            logger.error(f"ValueError in {context['active_submenu']}_view (POST): {ve}", exc_info=True)
        except Exception as e:
            context['error_message'] = f"처리 중 오류 발생: {str(e)}"
            logger.error(f"Unexpected error in {context['active_submenu']}_view (POST): {e}", exc_info=True)

    return render(request, 'routing_logistics_app/vrp_demo.html', context)


def cvrp_introduction_view(request):
    context = {
        'active_model': 'Routing & Logistics',  # 대메뉴 활성화용
        'active_submenu_category': 'capacitated_vehicle_routing_problems',
        'active_submenu': 'cvrp_introduction'  # 현재 페이지 활성화용
    }
    logger.debug(f"Rendering CVRP introduction page.")
    return render(request, 'routing_logistics_app/cvrp_introduction.html', context)


def cvrp_demo_view(request):
    """
    Capacitated Vehicle Routing Problem (CVRP) introduction page.
    """
    form_data ={}
    model_name='CVRP'
    if request.method == 'GET':
        logger.info(f'{model_name} Demo GET request processing.')
        submitted_num_customers = int(request.GET.get('num_customers_to_show', preset_num_customers))
        submitted_num_vehicles = int(request.GET.get('num_vehicles_to_show', preset_num_vehicles))
        submitted_vehicle_capa = int(request.GET.get('num_vehicle_capa_to_show', preset_vehicle_capacity))
        submitted_num_customers = max(1, min(10, submitted_num_customers))
        submitted_num_vehicles = max(1, min(5, submitted_num_vehicles))
        submitted_vehicle_capa = max(50, min(100, submitted_vehicle_capa))

        form_data['depot_x'] = preset_depot_location.get('x')
        form_data['depot_y'] = preset_depot_location.get('y')
        form_data['vehicle_capacity'] = preset_vehicle_capacity

        for i in range(len(preset_customer_locations)):
            preset = preset_customer_locations[i % len(preset_customer_locations)]
            for key, default_val in preset.items():
                form_data[f'cust_{i}_{key}'] = request.GET.get(f'cust_{i}_{key}', default_val)

    elif request.method == 'POST':
        form_data = request.POST.copy()
        submitted_num_customers = int(form_data.get('num_customers', preset_num_customers))
        submitted_num_vehicles = int(form_data.get('num_vehicles', preset_num_vehicles))
        submitted_vehicle_capa = int(form_data.get('vehicle_capa', preset_vehicle_capacity))

    context = {
        'active_model': 'Routing & Logistics',
        'active_submenu_category': 'capacitated_vehicle_routing_problems',
        'active_submenu': 'CVRP Demo',
        'form_data': form_data,
        'opt_results': None,
        'error_message': None, 'success_message': None,
        'processing_time_seconds': "N/A",
        'num_customers_options': range(1, 11),
        'num_vehicles_options': range(1, 6),
        'vehicle_capa_options': range(50, 210, 10),
        'submitted_num_customers': submitted_num_customers,
        'submitted_num_vehicles': submitted_num_vehicles,
        'submitted_vehicle_capa': submitted_vehicle_capa,
        'plot_data': None
    }

    if request.method == 'POST':
        logger.info(f'{model_name} Demo POST request processing.')

        try:
            # 1. 데이터 파일 새성 및 검증
            input_data = create_vrp_json_data(form_data)

            # 2. 파일 저장
            if settings.SAVE_DATA_FILE:
                success_save_message, save_error = save_vrp_json_data(input_data)
                if save_error:
                    context['error_message'] = (context.get('error_message', '') + " " + save_error).strip()
                elif success_save_message:
                    context['success_save_message'] = success_save_message

            # 3. 최적화 실행
            results_data, error_msg_opt, processing_time = run_cvrp_optimizer(input_data)
            context['processing_time_seconds'] = processing_time

            if error_msg_opt:
                context['error_message'] = (context.get('error_message', '') + " " + error_msg_opt).strip()
            elif results_data and (
                    results_data.get('routes') or results_data.get('total_distance') is not None):
                context['opt_results'] = results_data
                success_message=f"{model_name} 최적 경로 계산 완료! 총 거리: {results_data['total_distance']:.2f}"
                context['success_message'] = success_message
                logger.info(success_message)

                # 차트용 데이터 준비
                plot_data = {'locations': [], 'routes': [], 'depot_index': 0}
                # 모든 위치 (차고지 + 고객)
                depot_location = input_data.get('depot_location')
                customer_locations = input_data.get('customer_locations')

                plot_data['locations'].append({'id': 'Depot', 'x': depot_location['x'], 'y': depot_location['y']})
                for i, cust_loc in enumerate(customer_locations):
                    plot_data['locations'].append(
                        {'id': cust_loc['id'], 'x': cust_loc['x'], 'y': cust_loc['y'],
                         'demand': cust_loc.get('demand', 0)})

                # 경로 데이터 (좌표 시퀀스)
                for route_info in results_data['routes']:
                    plot_data['routes'].append({
                        'vehicle_id': route_info['vehicle_id'],
                        'path_coords': route_info['route_locations'],
                        'load': route_info.get('load', 0),  # 경로별 적재량 추가
                        'capacity': route_info.get('capacity', 0)  # 차량 용량 추가
                    })
                context['plot_data'] = json.dumps(plot_data)   # JSON 문자열로 전달
                logger.debug(f"Plot data prepared for {model_name}: {plot_data}")
            else:# 결과도 없고 명시적 에러도 없는 경우 (예: 해를 못 찾았지만 오류는 아닌 상태)
                current_error = context.get('error_message', '')
                default_opt_error = "최적 경로를 찾지 못했거나 방문할 고객이 없습니다."
                context['error_message'] = (
                            current_error + " " + default_opt_error).strip() if current_error else default_opt_error
        except ValueError as ve:
            context['error_message'] = f"입력값 오류: {str(ve)}"
            logger.error(f"ValueError in {context['active_submenu']}_view (POST): {ve}", exc_info=True)
        except Exception as e:
            context['error_message'] = f"처리 중 오류 발생: {str(e)}"
            logger.error(f"Unexpected error in {context['active_submenu']}_view (POST): {e}", exc_info=True)

    return render(request, 'routing_logistics_app/cvrp_demo.html', context)


def pdp_introduction_view(request):
    context = {
        'active_model': 'Routing & Logistics',  # 대메뉴 활성화용
        'active_submenu_category': 'pickup_delivery_problem',
        'active_submenu': 'pdp_introduction'  # 현재 페이지 활성화용
    }
    logger.debug("Rendering PDP introduction page.")
    return render(request, 'routing_logistics_app/pdp_introduction.html', context)


def pdp_demo_view(request):
    """
    Pickup and Delivery Problem 데모 뷰.
    """
    form_data = {}
    model_name = 'PDP'
    if request.method == 'GET':
        logger.info(f'{model_name} Demo GET request processing.')
        submitted_num_pairs = int(request.GET.get('num_pairs_to_show', preset_num_pairs))
        submitted_num_vehicles = int(request.GET.get('num_vehicles_to_show', preset_num_vehicles))
        submitted_vehicle_capa = int(request.GET.get('num_vehicle_capa_to_show', preset_vehicle_capacity))
        submitted_num_pairs = max(1, min(5, submitted_num_pairs))
        submitted_num_vehicles = max(1, min(5, submitted_num_vehicles))
        submitted_vehicle_capa = max(50, min(100, submitted_vehicle_capa))

        form_data['depot_x'] = preset_depot_location.get('x')
        form_data['depot_y'] = preset_depot_location.get('y')
        form_data['vehicle_capacity'] = preset_vehicle_capacity

        for i in range(len(preset_pair_locations)):
            preset = preset_pair_locations[i % len(preset_pair_locations)]
            for key, default_val in preset.items():
                form_data[f'pair_{i}_{key}'] = request.GET.get(f'pair_{i}_{key}', default_val)

    elif request.method == 'POST':
        form_data = request.POST.copy()
        submitted_num_pairs = int(form_data.get('num_pairs', preset_num_pairs))
        submitted_num_vehicles = int(form_data.get('num_vehicles', preset_num_vehicles))
        submitted_vehicle_capa = int(form_data.get('vehicle_capa', preset_vehicle_capacity))

    context = {
        'active_model': 'Routing & Logistics',
        'active_submenu_category': 'pickup_delivery_problems',
        'active_submenu': 'PDP Demo',
        'form_data': form_data,
        'opt_results': None,
        'error_message': None, 'success_message': None,
        'processing_time_seconds': "N/A",
        'num_pairs_options': range(1, 6),  # 1~5개 작업 쌍
        'num_vehicles_options': range(1, 6),
        'vehicle_capa_options': range(50, 210, 10),
        'submitted_num_pairs': submitted_num_pairs,
        'submitted_num_vehicles': submitted_num_vehicles,
        'submitted_vehicle_capa': submitted_vehicle_capa,
        'plot_data': None
    }

    if request.method == 'POST':
        logger.info("PDP Demo POST request processing.")
        try:
            # 1. 데이터 파일 새성 및 검증
            input_data = create_pdp_json_data(form_data)

            # 2. 파일 저장
            if settings.SAVE_DATA_FILE:
                success_save_message, save_error = save_vrp_json_data(input_data)
                if save_error:
                    context['error_message'] = (context.get('error_message', '') + " " + save_error).strip()
                elif success_save_message:
                    context['success_save_message'] = success_save_message

            # 3. 최적화 실행
            results_data, error_msg_opt, processing_time = run_pdp_optimizer(input_data)
            context['processing_time_seconds'] = processing_time

            if error_msg_opt:
                context['error_message'] = error_msg_opt
            elif results_data:
                context['opt_results'] = results_data
                context['success_message'] = f"PDP 최적 경로 계산 완료! 총 거리: {results_data.get('total_distance', 0):.2f}"

                # 차트용 데이터 준비
                plot_data = {'locations': [], 'routes': [], 'depot_index': 0, 'pairs': []}
                plot_data['locations'].append(
                    {'id': 'Depot', 'x': input_data['depot_location']['x'], 'y': input_data['depot_location']['y']})

                node_idx_counter = 1
                for i, pair in enumerate(input_data['pickup_delivery_pairs']):
                    plot_data['locations'].append(
                        {'id': f"{pair['id']}-P", 'x': pair['pickup']['x'], 'y': pair['pickup']['y']})
                    plot_data['locations'].append(
                        {'id': f"{pair['id']}-D", 'x': pair['delivery']['x'], 'y': pair['delivery']['y']})
                    plot_data['pairs'].append({'p_idx': node_idx_counter, 'd_idx': node_idx_counter + 1})
                    node_idx_counter += 2

                for route_info in results_data.get('routes', []):
                    plot_data['routes'].append({
                        'vehicle_id': route_info['vehicle_id'],
                        'path_coords': route_info['route_locations']
                    })
                context["locations_dict"] = {i: loc for i, loc in enumerate(plot_data["locations"])}
                context['plot_data'] = json.dumps(plot_data)
        except (ValueError, TypeError) as ve:
            context['error_message'] = f"입력값 오류: {str(ve)}"
            logger.error(f"ValueError in pdp_demo_view (POST): {ve}", exc_info=True)
        except Exception as e:
            context['error_message'] = f"처리 중 오류 발생: {str(e)}"
            logger.error(f"Unexpected error in pdp_demo_view (POST): {e}", exc_info=True)
    return render(request, 'routing_logistics_app/pdp_demo.html', context)


def vrp_test(request):
    context = {
        'active_model': 'Routing & Logistics',  # 대메뉴 활성화용
        'active_submenu_category': 'pickup_delivery_problem',
        'active_submenu': 'pdp_introduction'  # 현재 페이지 활성화용
    }
    return render(request, 'routing_logistics_app/vrp_test.html', context)