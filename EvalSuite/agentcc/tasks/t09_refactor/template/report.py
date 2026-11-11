def line_item(name, amount):
    return f'{name}: ${amount:,.2f}'


def total_line(amount):
    return f'Total: ${amount:,.2f}'


def tax_line(amount, rate):
    return f'Tax: ${amount * rate:,.2f}'
