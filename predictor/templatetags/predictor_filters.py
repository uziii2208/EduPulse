from django import template

register = template.Library()

@register.filter
def multiply(value, arg):
    """Multiplies the arg and value"""
    try:
        return float(value) * float(arg)
    except (ValueError, TypeError):
        return ''