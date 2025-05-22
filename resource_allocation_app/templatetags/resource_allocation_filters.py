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
#         {{ form_data|get_item:item_name_key }}
#     {% endwith %}
# 아래는 get_item을 위한 간단한 버전. 템플릿의 printf는 제거.

@register.filter
def get_item_simple(dictionary, key):
    if isinstance(dictionary, dict):
        return dictionary.get(key)
    return None

# floatformat, default_if_none, add 등은 Django 내장 필터 사용.