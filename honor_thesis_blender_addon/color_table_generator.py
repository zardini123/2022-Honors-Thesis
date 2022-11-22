import sys

from . import non_dependent_utilities


def main() -> int:
    for i in range(0, 100):
        print(non_dependent_utilities.unique_color_from_number(i))

    return 0


if __name__ == '__main__':
    sys.exit(main())
