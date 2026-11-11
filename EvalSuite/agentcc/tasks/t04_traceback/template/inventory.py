def total_value(items):
    return sum(i["price"] * i["qty"] for i in items)


def load_items():
    return [{"price": 2.5, "qty": 4}, {"price": 10.0}, {"price": 1.0, "qty": 3}]
