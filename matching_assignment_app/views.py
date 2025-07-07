from django.shortcuts import render
from django.conf import  settings

from common_utils.run_matching_opt import *
from common_utils.data_utils_matching import *

import os
import json
import logging


logger = logging.getLogger(__name__)

def validate_panel_data_structure(panel_list_items, panel_type_name):
    """
    패널 데이터 리스트의 구조와 내용의 유효성을 검사합니다.
    오류가 있으면 오류 메시지 문자열을, 정상이면 None을 반환합니다.
    """
    if panel_list_items is None:
        return f"오류: '{panel_type_name}_panels' 데이터가 없습니다 (None)."
    if not isinstance(panel_list_items, list):
        return f"오류: '{panel_type_name}_panels' 데이터가 리스트 형식이 아닙니다."
    if not panel_list_items:  # 빈 리스트도 유효할 수 있으나, 여기서는 패널이 있어야 한다고 가정
        return f"오류: '{panel_type_name}_panels' 리스트가 비어있습니다."

    for p_idx, p_item in enumerate(panel_list_items):
        if not isinstance(p_item, dict):
            return f"오류: {panel_type_name} 패널 데이터 (인덱스 {p_idx})가 딕셔너리 형식이 아닙니다."

        required_keys = ('id', 'rows', 'cols', 'defect_map')
        missing_keys = [k for k in required_keys if k not in p_item]
        if missing_keys:
            return f"오류: {panel_type_name} 패널 (ID: {p_item.get('id', 'N/A')}, 인덱스 {p_idx})에 필수 키가 누락되었습니다: {', '.join(missing_keys)}."

        panel_id = p_item.get('id', f'인덱스 {p_idx}')
        rows = p_item.get('rows')
        cols = p_item.get('cols')
        defect_map = p_item.get('defect_map')

        if not (isinstance(rows, int) and rows > 0 and isinstance(cols, int) and cols > 0):
            return f"오류: {panel_type_name} 패널 {panel_id}의 rows/cols 값이 유효한 양의 정수가 아닙니다 (rows: {rows}, cols: {cols})."

        if not isinstance(defect_map, list) or len(defect_map) != rows:
            return f"오류: {panel_type_name} 패널 {panel_id}의 defect_map 형식이 잘못되었거나 행 수가 명시된 rows({rows})와 일치하지 않습니다 (실제 행 수: {len(defect_map) if isinstance(defect_map, list) else 'N/A'})."

        for r_idx, row_data in enumerate(defect_map):
            if not isinstance(row_data, list) or len(row_data) != cols:
                return f"오류: {panel_type_name} 패널 {panel_id}의 defect_map의 행 {r_idx}의 열 수가 명시된 cols({cols})와 일치하지 않습니다 (실제 열 수: {len(row_data) if isinstance(row_data, list) else 'N/A'})."
            if not all(cell_val in (0, 1) for cell_val in row_data):
                return f"오류: {panel_type_name} 패널 {panel_id}의 defect_map에 유효하지 않은 값(0 또는 1이 아님)이 포함되어 있습니다 (행 {r_idx})."
    return None  # 유효성 검사 통과


def matching_assignment_introduction_view(request):
    """General introduction page for the Matching & Assignment category."""
    context = {
        'active_model': 'Matching & Assignment',
        'active_submenu': 'main_introduction'
    }
    logger.debug("Rendering general Matching & Assignment introduction page.")
    return render(request, 'matching_assignment_app/matching_assignment_introduction.html', context)


def lcd_cf_tft_introduction_view(request):
    context = {
        'active_model': 'Matching & Assignment',
        'active_submenu': 'introduction',
    }
    return render(request, 'matching_assignment_app/lcd_cf_tft_introduction.html', context)

def lcd_cf_tft_data_generation_view(request):
    context = {
        'active_model': 'Matching & Assignment',
        'active_submenu_category': 'lcd_tft_cf_matching',
        'active_submenu': 'lcd_tft_cf_data_generation',
        'cf_tft_panel_range': range(3, 11),
        'cell_dimension_range': range(3, 6),
        'form_values': request.POST if request.method == 'POST' else {},  # GET 요청 시 빈 dict
    }

    if request.method == 'POST':
        logger.debug(f"Data generation POST request received. Data: {request.POST}")
        try:
            num_cf_panels = int(request.POST.get('num_cf_panels', 3))
            num_tft_panels = int(request.POST.get('num_tft_panels', 3))
            panel_rows = int(request.POST.get('panel_rows', 3))
            panel_cols = int(request.POST.get('panel_cols', 3))
            defect_rate = int(request.POST.get('defect_rate', 10))

            generated_data = create_cf_tft_matching_json_data(num_cf_panels, num_tft_panels, panel_rows, panel_cols, defect_rate)
            context['generated_data'] = generated_data
            generated_data_json_pretty = json.dumps(generated_data, indent=4)
            context['generated_data_json_pretty'] = generated_data_json_pretty
            request.session['generated_lcd_data'] = generated_data_json_pretty
            logger.info("Panel data generated successfully.")
        except ValueError as e:
            context['error_message'] = "잘못된 입력입니다. 모든 숫자가 올바르게 입력되었는지 확인하세요."
            logger.error(f"ValueError during data generation: {e}", exc_info=True)
        except Exception as e:
            context['error_message'] = f"데이터 생성 중 오류 발생: {str(e)}"
            logger.error(f"Unexpected error during data generation: {e}", exc_info=True)

    return render(request, 'matching_assignment_app/lcd_cf_tft_data_generation.html', context)


def lcd_cf_tft_small_scale_demo_view(request):
    context = {
        'active_model': 'Matching & Assignment',
        'active_submenu_category': 'lcd_tft_cf_matching',
        'active_submenu': 'lcd_cf_tft_small_scale_demo'
    }

    if 'generated_lcd_data' in request.session:
        # 세션에 데이터가 있으면 가져와서 context에 추가
        generated_data = request.session['generated_lcd_data']
        context['submitted_json_data'] = generated_data
        # 사용 후 세션 데이터 삭제 (새로고침 시 재사용 방지)
        del request.session['generated_lcd_data']
        logger.info("Loaded generated data from session.")

    if request.method == 'POST':
        test_data_json_str = request.POST.get('test_data_json')
        context['submitted_json_data'] = test_data_json_str
        logger.info("Small scale demo POST request received.")
        logger.debug(f"Submitted JSON data: {test_data_json_str[:200]}...")  # 너무 길면 일부만 로깅

        if test_data_json_str:
            try:
                test_data = json.loads(test_data_json_str)
                cf_panels = test_data.get('cf_panels')
                tft_panels = test_data.get('tft_panels')

                # 공통 유효성 검사 함수 사용
                validation_error_cf = validate_panel_data_structure(cf_panels, "CF")
                if validation_error_cf:
                    logger.error(f"CF Panel Validation Error: {validation_error_cf}")
                    context['error_message'] = validation_error_cf
                    return render(request, 'matching_assignment_app/lcd_cf_tft_small_scale_demo.html', context)

                validation_error_tft = validate_panel_data_structure(tft_panels, "TFT")
                if validation_error_tft:
                    logger.error(f"TFT Panel Validation Error: {validation_error_tft}")
                    context['error_message'] = validation_error_tft
                    return render(request, 'matching_assignment_app/lcd_cf_tft_small_scale_demo.html', context)

                # 유효성 검사 통과 후
                context['input_cf_panels'] = cf_panels
                context['input_tft_panels'] = tft_panels
                logger.info("Input panel data validated successfully.")

                matched_pairs, total_yield, error_msg, solver_time = run_matching_cf_tft_algorithm(cf_panels, tft_panels)

                if error_msg:
                    context['error_message'] = error_msg
                    logger.error(f"Error from matching algorithm: {error_msg}")
                else:
                    context['matching_pairs'] = matched_pairs
                    context['total_yield'] = total_yield
                    if matched_pairs or total_yield > 0:
                        msg = f"매칭 완료! 총 수율: {total_yield:.0f}"
                        context['success_message'] = msg
                        logger.info(msg)
                    elif not error_msg:  # 에러 없고 매칭 결과도 없을 때
                        msg = "매칭 가능한 쌍이 없거나 모든 쌍의 수율이 0입니다."
                        context['success_message'] = msg  # 정보성 메시지로 처리
                        logger.info(msg)

            except json.JSONDecodeError as e:
                msg = "오류: 잘못된 JSON 형식입니다."
                logger.error(f"{msg} - {e}", exc_info=True)
                context['error_message'] = msg
            except ValueError as ve:  # 직접 발생시킨 ValueError 포함
                msg = f"데이터 유효성 검사 또는 처리 오류: {str(ve)}"
                logger.error(msg, exc_info=True)
                context['error_message'] = msg
            except Exception as e:
                msg = f"매칭 중 예상치 못한 오류 발생: {str(e)}"
                logger.error(msg, exc_info=True)
                context['error_message'] = msg
        else:
            context['error_message'] = "오류: 테스트 데이터가 제공되지 않았습니다."
            logger.warning("No test data provided for small scale demo.")

    return render(request, 'matching_assignment_app/lcd_cf_tft_small_scale_demo.html', context)


def lcd_cf_tft_large_scale_demo_view(request):
    context = {
        'active_model': 'Matching & Assignment',
        'active_submenu_category': 'lcd_tft_cf_matching',
        'active_submenu': 'lcd_cf_tft_large_scale_demo',
        'available_json_files': [],
        'is_cloud': getattr(settings, 'USE_GCS', False),
    }

    data_dir_path_str = settings.DEMO_DIR_MAP['matching_cf_tft_data']
    if data_dir_path_str and os.path.isdir(data_dir_path_str):
        try:
            files = [f for f in os.listdir(data_dir_path_str) if f.endswith('.json') and f.startswith('cf')]
            logger.info(f"DIR:{data_dir_path_str}, available_json_files:{len(files)}")
            context['available_json_files'] = [{'value': f, 'name': f} for f in sorted(files, reverse=True)]
        except OSError as e:
            logger.error(f"Error listing files in {data_dir_path_str}: {e}")
            context['error_message'] = f"서버의 데이터 디렉토리에서 파일 목록을 읽어오는 데 실패했습니다."
    elif not data_dir_path_str:
        logger.warning(f"{data_dir_path_str} is not defined in settings. File selection will not work.")
        # context['error_message'] = "서버 데이터 디렉토리가 설정되지 않았습니다." # 사용자에게 보여줄 필요는 없을 수도 있음

    if request.method == 'POST':
        logger.info(f"Large scale demo POST request received. Input type: {request.POST.get('large_data_input_type')}")
        input_type = request.POST.get('large_data_input_type')
        cf_panels = None
        tft_panels = None
        loaded_filename = None

        try:
            if input_type == 'make_json':
                logger.info("Processing 'make_json' input type.")
                num_cf = request.POST.get('num_cf_panels', '100')
                num_tft = request.POST.get('num_tft_panels', '100')
                panel_r = request.POST.get('panel_rows', '4')
                panel_c = request.POST.get('panel_cols', '4')
                defect_rate_str = request.POST.get('defect_rate', '10')

                # 입력값 유효성 검사 (숫자 변환 및 범위)
                num_cf = int(num_cf)
                num_tft = int(num_tft)
                panel_r = int(panel_r)
                panel_c = int(panel_c)
                defect_rate_percent  = int(defect_rate_str)

                generated_data = create_cf_tft_matching_json_data(num_cf, num_tft, panel_r, panel_c, defect_rate_percent)
                if data_dir_path_str:
                    # 중복 방지를 위해 시퀀스 번호 또는 타임스탬프 사용
                    seq = 0
                    cf_panels = generated_data.get('cf_panels')
                    tft_panels = generated_data.get('tft_panels')

                    while True:
                        filename_pattern = f"cf{num_cf}_tft{num_tft}_row{panel_r}_col{panel_c}_rate{str(defect_rate_percent ).replace('.', 'p')}"
                        if seq == 0:
                            potential_filename = f"{filename_pattern}.json"
                        else:
                            potential_filename = f"{filename_pattern}_seq{seq}.json"

                        loaded_filename = potential_filename

                        filepath = os.path.join(data_dir_path_str, potential_filename)
                        if not os.path.exists(filepath):
                            loaded_filename = potential_filename  # 저장될 (또는 사용될) 파일명
                            with open(filepath, 'w', encoding='utf-8') as f:
                                json.dump(generated_data, f, indent=2)
                            logger.info(f"Generated data saved to: {filepath}")
                            context['success_message'] = f"데이터가 생성되어 '{loaded_filename}'으로 서버에 저장되었습니다. 이제 매칭을 실행합니다."
                            # 파일 목록을 즉시 업데이트하기 위해 다시 로드 (선택 사항)
                            files = [f for f in os.listdir(data_dir_path_str) if
                                     f.endswith('.json') and f.startswith('cf')]
                            context['available_json_files'] = [{'value': f, 'name': f} for f in
                                                               sorted(files, reverse=True)]
                            break
                        seq += 1
                        if seq > 100:  # 무한 루프 방지
                            logger.error("Could not find a unique filename after 100 attempts for make_json.")
                            context['error_message'] = "생성된 데이터를 저장할 고유한 파일 이름을 찾는 데 실패했습니다."
                            return render(request, 'matching_assignment_app/lcd_cf_tft_large_scale_demo.html', context)
                else:
                    logger.warning("MATCH_CF_TFT_DATA_DIR not set. Generated data will not be saved.")
                    context['info_message'] = "데이터가 생성되었지만, 서버 저장 경로가 설정되지 않아 저장되지 않았습니다. 매칭은 진행됩니다."


            elif input_type == 'select_json':
                logger.info("Processing 'select_json' input type.")
                selected_file = request.POST.get('selected_json_file')
                if not selected_file:
                    context['error_message'] = "서버에서 JSON 파일이 선택되지 않았습니다."
                elif not data_dir_path_str:
                    context['error_message'] = "서버 데이터 디렉토리가 설정되지 않아 파일을 로드할 수 없습니다."
                else:
                    filepath = os.path.join(data_dir_path_str, selected_file)
                    if os.path.exists(filepath):
                        with open(filepath, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        cf_panels = data.get('cf_panels')
                        tft_panels = data.get('tft_panels')
                        loaded_filename = selected_file
                        logger.info(f"Data loaded from selected server file: {filepath}")
                    else:
                        context['error_message'] = f"선택한 파일 '{selected_file}'을 서버에서 찾을 수 없습니다."

            elif input_type == 'upload_json':
                logger.info("Processing 'upload_json' input type.")
                uploaded_file = request.FILES.get('data_file')
                if not uploaded_file:
                    context['error_message'] = "업로드된 JSON 파일이 없습니다."
                elif not uploaded_file.name.endswith('.json'):
                    context['error_message'] = "잘못된 파일 형식입니다. JSON 파일만 업로드 가능합니다."
                else:
                    try:
                        # FileSystemStorage를 사용하면 임시 파일 또는 메모리에서 바로 처리 가능
                        # fs = FileSystemStorage()
                        # filename = fs.save(uploaded_file.name, uploaded_file) # 임시 저장 (선택)
                        # filepath = fs.path(filename)
                        # with open(filepath, 'r', encoding='utf-8') as f:
                        #     data = json.load(f)
                        # fs.delete(filename) # 임시 파일 삭제

                        # 메모리에서 직접 읽기 (더 효율적)
                        data = json.load(uploaded_file)
                        cf_panels = data.get('cf_panels')
                        tft_panels = data.get('tft_panels')
                        loaded_filename = uploaded_file.name
                        logger.info(f"Data loaded from uploaded file: {uploaded_file.name}")
                    except json.JSONDecodeError:
                        context['error_message'] = "업로드된 JSON 파일의 형식이 올바르지 않습니다."
                    except Exception as e:
                        context['error_message'] = f"파일 처리 중 오류 발생: {str(e)}"
                        logger.error(
                            f"Error processing uploaded file {uploaded_file.name if uploaded_file else 'N/A'}: {e}",
                            exc_info=True)
            else:
                context['error_message'] = "알 수 없는 입력 유형입니다."

            # --- 데이터 로드 또는 생성 후 유효성 검사 및 매칭 실행 ---
            if cf_panels is not None and tft_panels is not None:
                validation_error_cf = validate_panel_data_structure(cf_panels, "CF")
                if validation_error_cf:
                    logger.error(f"Large Scale CF Panel Validation Error: {validation_error_cf}")
                    context['error_message'] = validation_error_cf
                    return render(request, 'matching_assignment_app/lcd_cf_tft_large_scale_demo.html', context)

                validation_error_tft = validate_panel_data_structure(tft_panels, "TFT")
                if validation_error_tft:
                    logger.error(f"Large Scale TFT Panel Validation Error: {validation_error_tft}")
                    context['error_message'] = validation_error_tft
                    return render(request, 'matching_assignment_app/lcd_cf_tft_large_scale_demo.html', context)

                logger.info(
                    f"Data for large scale matching validated. CF: {len(cf_panels)}, TFT: {len(tft_panels)}. Source: {loaded_filename}")

                matched_pairs, total_yield, error_msg, solver_time = run_matching_cf_tft_algorithm(cf_panels, tft_panels)

                if error_msg:
                    context['error_message'] = (context.get('error_message', '') + " " + error_msg).strip()
                else:
                    num_matches = len(matched_pairs)
                    avg_yield = total_yield / num_matches if num_matches > 0 else 0
                    processing_time_val=solver_time
                    context['large_scale_results'] = {
                        'num_cf': len(cf_panels),
                        'num_tft': len(tft_panels),
                        'num_matches': num_matches,
                        'total_yield': round(total_yield),
                        'avg_yield': avg_yield,
                        'processing_time_seconds': processing_time_val,
                        'sample_matches': matched_pairs[:10],  # 처음 10개만 샘플로
                        'source_file': loaded_filename if loaded_filename else "Newly Generated (unsaved or error saving)"
                    }
                    success_msg_main = f"대규모 매칭 완료 (소스: {loaded_filename})."
                    current_success = context.get('success_message_extra', '')  # 파일 저장 성공 메시지 등
                    context['success_message'] = (
                                current_success + " " + success_msg_main).strip() if current_success else success_msg_main
                    logger.info(
                        f"Large scale matching completed. Total yield: {total_yield}. Source: {loaded_filename}")


            elif not context.get('error_message'):  # 데이터 로드/생성 실패했고, 명시적 에러 메시지 없을 때
                context['error_message'] = "패널 데이터를 준비하지 못했습니다."
                logger.warning(
                    "Panel data could not be prepared for large scale matching and no specific error was set.")


        except Exception as e:
            context['error_message'] = f"처리 중 오류 발생: {str(e)}"
            logger.error(f"Unexpected error in large_scale_demo_view: {e}", exc_info=True)

    return render(request, 'matching_assignment_app/lcd_cf_tft_large_scale_demo.html', context)


def assignment_introduction_view(request):
    context = {
        'active_model': 'Matching & Assignment',
        'active_submenu_category': 'assignment_problems',
        'active_submenu': 'assignment_introduction'
    }
    logger.debug("Rendering Assignment Problem introduction page.")
    return render(request, 'matching_assignment_app/assignment_introduction.html', context)


def transport_assignment_introduction_view(request):
    """
    Transportation Assignment Problem Introduction Page.
    """
    context = {
        'active_model': 'Matching & Assignment',
        'active_submenu_category': 'transport_assignment_problems',
        'active_submenu': 'transport_assignment_introduction'
    }
    logger.debug("Rendering Transportation Assignment introduction page.")
    return render(request, 'matching_assignment_app/transport_assignment_introduction.html', context)


def transport_assignment_demo_view(request):
    """
        Transportation Assignment Problem 데모 뷰.
        """
    form_data = {}

    if request.method == 'GET':
        submitted_num_items = int(request.GET.get('num_items_to_show', preset_trans_assign_items))
        submitted_num_items = max(2, min(5, submitted_num_items))  # 2~5개로 제한

        # GET 요청 시 랜덤 비용 행렬로 form_data 초기화
        for i in range(submitted_num_items):
            # URL 파라미터가 있으면 그 값을, 없으면 기본값을 사용
            form_data[f'driver_name_{i}'] = request.GET.get(f'driver_name_{i}', preset_trans_assign_drivers[i])
            form_data[f'zone_name_{i}'] = request.GET.get(f'zone_name_{i}', preset_trans_assign_zones[i])
            for j in range(submitted_num_items):
                cost_key = f'cost_{i}_{j}'
                form_data[cost_key] = request.GET.get(cost_key, str(random.randint(20, 100)))

    elif request.method == 'POST':
        form_data = request.POST.copy()
        submitted_num_items = int(form_data.get('num_items', preset_trans_assign_items))

    context = {
        'active_model': 'Matching & Assignment',
        'active_submenu_category': 'transport_assignment_problems',
        'active_submenu': 'Transport Assignment Demo',
        'form_data': form_data,
        'assignment_results': None,
        'error_message': None, 'success_message': None,
        'processing_time_seconds': "N/A",
        'num_items_options': range(2, 6),  # 2x2 ~ 5x5 행렬
        'submitted_num_items': submitted_num_items
    }

    if request.method == 'POST':
        logger.info("Transportation Assignment Demo POST request processing.")
        try:
            # 1. 데이터 파일 새성 및 검증
            input_data =create_transport_assignment_json_data(form_data, submitted_num_items)

            # 2. 파일 저장
            if settings.SAVE_DATA_FILE:
                success_save_message, save_error = save_matching_assignment_json_data(input_data)
                if save_error:
                    context['error_message'] = (context.get('error_message', '') + " " + save_error).strip()  # 기존 에러에 추가
                elif success_save_message:
                    context['success_save_message'] = success_save_message

            # 3. 최적화 실행
            results_data, error_msg_opt, processing_time = run_matching_transport_optimizer(input_data)
            context['processing_time_seconds'] = processing_time

            if error_msg_opt:
                context['error_message'] = error_msg_opt
            elif results_data:
                context['assignment_results'] = results_data
                context['success_message'] = f"최적 할당 완료! 최소 총 비용(시간): {results_data['total_cost']}"
                logger.info(f"Assignment successful. Total cost: {results_data['total_cost']}")
            else:
                context['error_message'] = "최적 할당 결과를 가져오지 못했습니다."

        except ValueError as ve:
            context['error_message'] = f"입력값 오류: {str(ve)}"
            logger.error(f"ValueError in transport_assignment_demo_view: {ve}", exc_info=True)
        except Exception as e:
            context['error_message'] = f"처리 중 오류 발생: {str(e)}"
            logger.error(f"Unexpected error in transport_assignment_demo_view: {e}", exc_info=True)

    return render(request, 'matching_assignment_app/transport_assignment_demo.html', context)


def resource_skill_matching_introduction_view(request):
    """
    Resource-Skill Matching Problem Introduction Page.
    """
    context = {
        'active_model': 'Matching & Assignment',
        'active_submenu_category': 'resource_skill_matching_problems',
        'active_submenu': 'resource_skill_matching_introduction'
    }
    logger.debug("Rendering Resource-Skill Matching introduction page.")
    return render(request, 'matching_assignment_app/resource_skill_matching_introduction.html', context)


def resource_skill_matching_demo_view(request):
    """
    Resource-Skill Matching 데모 뷰.
    """
    form_data = {}

    # GET 요청 시: URL 파라미터 또는 기본값으로 항목 수 결정
    if request.method == 'GET':
        submitted_num_resources = int(request.GET.get('num_resources_to_show', preset_num_resources))
        submitted_num_projects = int(request.GET.get('num_projects_to_show', preset_num_projects))
        submitted_num_resources = max(1, min(10, submitted_num_resources))
        submitted_num_projects = max(1, min(5, submitted_num_projects))

        # 선택된 수만큼 form_data 채우기
        for i in range(submitted_num_resources):
            preset = preset_resources[i % len(preset_resources)]
            for key, default_val in preset.items():
                form_data[f'res_{i}_{key}'] = request.GET.get(f'res_{i}_{key}', default_val)
        for i in range(submitted_num_projects):
            preset = preset_projects[i % len(preset_resources)]
            for key, default_val in preset.items():
                form_data[f'proj_{i}_{key}'] = request.GET.get(f'proj_{i}_{key}', default_val)
    elif request.method == 'POST':
        form_data = request.POST.copy()
        submitted_num_resources = int(form_data.get('num_resources', preset_num_resources))
        submitted_num_projects = int(form_data.get('num_projects', preset_num_projects))

    context = {
        'active_model': 'Matching & Assignment',
        'active_submenu_category': 'resource_skill_matching_problems',
        'active_submenu': 'Resource Skill Matching Demo',
        'form_data': form_data,
        'results': None, 'error_message': None, 'success_message': None,
        'processing_time_seconds': "N/A",
        'num_resources_options': range(1, 11),  # 1~10명 인력
        'num_projects_options': range(1, 6),  # 1~5개 프로젝트
        'submitted_num_resources': submitted_num_resources,
        'submitted_num_projects': submitted_num_projects,
    }

    if request.method == 'POST':
        logger.info("Resource-Skill Matching Demo POST request processing.")
        try:
            # 1. 데이터 파일 새성 및 검증
            input_data = create_resource_skill_matching_json_data(form_data, submitted_num_resources, submitted_num_projects)

            # 2. 파일 저장
            if settings.SAVE_DATA_FILE:
                success_save_message, save_error = save_matching_assignment_json_data(input_data)
                if save_error:
                    context['error_message'] = (context.get('error_message', '') + " " + save_error).strip()  # 기존 에러에 추가
                elif success_save_message:
                    context['success_save_message'] = success_save_message
            # solving 단계에서 다양한 케이스 탐색 가능하여 주석 처리
            # unmatched, formatted_html = validate_required_skills(input_data)
            # if unmatched:
            #     from django.utils.safestring import mark_safe
            #     context['error_message'] = mark_safe(
            #         f"Cannot solve the problem: No available resources possess all the required skills for the project(s):<br>{formatted_html}"
            #     )
            #     logger.error(f"Validation error in resource-skill matching demo. Raw data: {unmatched}")
            #     return render(request, 'matching_assignment_app/resource_skill_matching_demo.html', context)

            # 3. 최적화 실행
            results_data, error_msg_opt, processing_time = run_skill_matching_optimizer(input_data)
            context['processing_time_seconds'] = processing_time

            if error_msg_opt:
                context['error_message'] = error_msg_opt
            elif results_data:
                context['results'] = results_data
                context['success_message'] = f"최적 팀 구성 완료! 최소 총 투입 비용: {results_data.get('total_cost', 0)}"
                logger.info(f"Skill matching successful. Total cost: {results_data.get('total_cost')}")
            else:
                context['error_message'] = "최적 할당 결과를 가져오지 못했습니다."

        except ValueError as ve:
            context['error_message'] = f"입력값 오류: {str(ve)}"
            logger.error(f"ValueError in skill matching demo: {ve}", exc_info=True)
        except Exception as e:
            context['error_message'] = f"처리 중 오류 발생: {str(e)}"
            logger.error(f"Unexpected error in skill matching demo: {e}", exc_info=True)

    return render(request, 'matching_assignment_app/resource_skill_matching_demo.html', context)
