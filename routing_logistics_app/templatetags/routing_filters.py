from django import template
import logging
logger = logging.getLogger(__name__)
register = template.Library()

@register.filter(name='get_range')
def get_range(number):
    # logger.debug(f"get_range: number = {number} (type: {type(number)})")
    if isinstance(number, int) and number >= 0:
        # logger.debug(f"get_range.isinstance: returning range({number})")
        return range(number)
    logger.warning(f"get_range: received invalid number '{number}', returning empty list.")
    return []


@register.filter
def get_item(dictionary, key_or_printf_args):
    """
    딕셔너리에서 키로 값을 가져옵니다.
    key_or_printf_args가 튜플이면 printf처럼 포맷팅된 키를 사용합니다.
    """
    if isinstance(key_or_printf_args, tuple) and len(key_or_printf_args) > 1:
        # printf 스타일 (format_string, arg1, arg2, ...)
        key = key_or_printf_args[0] % key_or_printf_args[1:]
    else:
        key = key_or_printf_args

    if isinstance(dictionary, dict):
        return dictionary.get(key)
    return None

# printf와 유사한 기능을 위한 필터. Django 템플릿에서는 직접적인 문자열 포맷팅이 제한적이므로,
# 뷰에서 key를 미리 만들거나, 이와 같은 필터를 사용.
# 이 필터는 value가 포맷 문자열, arg가 포맷 인자가 됩니다.
# 예: {{ "item_name_%s"|printf:i }}
# 하지만 위 템플릿에서는 get_item 필터 내에서 처리하도록 로직을 약간 변경함.
# 더 간단하게는, 뷰에서 form_values를 만들 때 item_0_name, item_1_name 등으로 만들고
# 템플릿에서는 item.name_key 와 같이 접근할 수 있도록 context 구조를 바꿀 수도 있음.
# 일단은 get_item 필터에서 printf 스타일을 흉내내도록 했습니다.
# get_item 호출 시 printf('item_name_%s', i)는 템플릿에서 직접 안되므로,
# 뷰에서 form_data를 처리할 때, 혹은 템플릿에서 좀 더 명시적으로 키를 만들어야 합니다.
#
# 더 나은 방법: 템플릿에서 문자열 결합은 add 필터를 사용
# 예: {% with item_name_key="item_name_"|add:i|stringformat:"s" %}
# 예: {% with key_name='total_space_sqm' %}{{ form_data|get_item_simple:key_name|default_if_none:'10' }}{% endwith %}
#         {{ form_data|get_item:item_name_key }}
#     {% endwith %}
# 아래는 get_item을 위한 간단한 버전. 템플릿의 printf는 제거.

@register.filter
def get_item_simple(collection, key):
    if isinstance(collection, dict):
        return collection.get(key)
    if isinstance(collection, list):
        try:
            # key가 정수형 인덱스로 변환 가능한지 확인
            int_key = int(key)
            logger.debug(f"get_item_simple: Trying to access list with index {int_key}")

            # 인덱스가 유효한 범위 내에 있는지 확인
            if 0 <= int_key < len(collection):
                item = collection[int_key]
                logger.debug(f"get_item_simple: Success! Returning item: {item}")
                return item
            else:
                logger.warning(f"get_item_simple: Index {int_key} is out of bounds for list of size {len(collection)}.")
                return None

        except (ValueError, TypeError) as e:
            # key를 정수로 변환할 수 없거나 다른 타입 에러 발생 시
            logger.error(f"get_item_simple: Failed to convert key '{key}' to int or other type error. Error: {e}")
            return None
        except IndexError as e:
            # 인덱스 에러 발생 시 (위의 범위 체크로 대부분 방지되지만, 만약을 위해)
            logger.error(f"get_item_simple: IndexError for key '{key}'. Error: {e}")
            return None
    return None

@register.simple_tag(name='make_key')
def make_key(prefix, index, suffix):
    return f"{prefix}_{index}_{suffix}"


# floatformat, default_if_none, add 등은 Django 내장 필터 사용.