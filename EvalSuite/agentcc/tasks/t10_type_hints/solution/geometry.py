import math


def circle_area(r: float) -> float:
    return math.pi * r * r


def rect_area(w: float, h: float) -> float:
    return w * h


def hypot(a: float, b: float) -> float:
    return math.sqrt(a * a + b * b)
