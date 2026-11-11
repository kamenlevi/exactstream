def format_money(amount):
    return f'${amount:,.2f}'


def line_item(name, amount):
    return f'{name}: {format_money(amount)}'


def total_line(amount):
    return f'Total: {format_money(amount)}'


def tax_line(amount, rate):
    return f'Tax: {format_money(amount * rate)}'
