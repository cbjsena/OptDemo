from django import template

register = template.Library()


@register.filter
def get_item(container, key):
    """딕셔너리에서 key로 값을 가져오거나, 리스트에서 인덱스로 값을 가져오는 필터"""
    if isinstance(container, dict):
        return container.get(key, container.get(str(key), 0))
    if isinstance(container, (list, tuple)):
        try:
            return container[int(key)]
        except (IndexError, ValueError, TypeError):
            return 0
    return 0

