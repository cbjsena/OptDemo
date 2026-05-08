from django import template

register = template.Library()


@register.filter
def get_item(dictionary, key):
    """딕셔너리에서 key로 값을 가져오는 필터"""
    if isinstance(dictionary, dict):
        return dictionary.get(key, dictionary.get(str(key), 0))
    return 0

