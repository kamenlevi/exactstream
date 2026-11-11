from shop.pricing import price_of


def cart_total(items):
    return sum(price_of(i) for i in items)
