def unique_color_from_number(number: float) -> tuple[float, float, float, float]:
    # 1/pi
    r = (number * 0.318309) % 1.0
    # 1/golden ratio
    g = (number * 0.61804) % 1.0
    # 1/sqrt(2)
    b = (number * 0.7071067812) % 1.0

    return (r, g, b, 1.0)
