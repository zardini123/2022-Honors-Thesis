import sys
import itertools


def unique_color_from_number(number: float) -> tuple[float, float, float, float]:
    # 1/pi
    r = (number * 0.318309) % 1.0
    # 1/golden ratio
    g = (number * 0.61804) % 1.0
    # 1/sqrt(2)
    b = (number * 0.7071067812) % 1.0

    return (r, g, b, 1.0)


def deviating_color(has_ridge: bool, min_deviating: bool, max_deviating: bool):
    r = g = b = 0.0

    # if self.has_ridge:
    #     g = 1.0
    # if self.near_umbilic:
    #     b = 1.0

    # if near_umbilic:
    #     r = 1.0

    if has_ridge:
        g = 1.0

    # Max is not deviating
    if min_deviating:
        b = 1.0

    if max_deviating:
        r = 1.0

    return (r, g, b, 1.0)


def main() -> int:
    colors = []
    tables = []

    def define_color(color_identifier, color_tuple):
        return f"\\definecolor{{{color_identifier}}}{{rgb}}{{{color_tuple[0]}, {color_tuple[1]}, {color_tuple[2]}}}"

    base_name = "zone_depth_"
    for i in range(0, 15):
        color_tuple = unique_color_from_number(i)
        color_identifier = base_name + str(i)
        colors.append(define_color(color_identifier, color_tuple))

        zone_width = 1 / (2 ** i)

        if zone_width > 0.01:
            zone_width_string = f"{zone_width:.4}"
        else:
            zone_width_string = f"{zone_width:.3e}"

        tables.append(f"{i} & \\boxofcolor{{{color_identifier}}} & {zone_width_string} \\\\")

    # Boolean SAT-like
    base_name = "zone_type_"

    def bool_tuple_to_string(vals):
        out = ""
        for val in vals:
            out += "1" if val else "0"
        return out

    for assignment in itertools.product([True, False], repeat=3):
        has_ridge, min_deviating, max_deviating = assignment

        color_tuple = deviating_color(has_ridge, min_deviating, max_deviating)
        color_identifier = base_name + bool_tuple_to_string(assignment)

        colors.append(define_color(color_identifier, color_tuple))

        tables.append(
            f"\\boxofcolor{{{color_identifier}}} & {has_ridge} & {min_deviating} & {max_deviating} \\\\"
        )

    out = colors + tables

    print("\n".join(out))

    return 0


if __name__ == '__main__':
    sys.exit(main())
