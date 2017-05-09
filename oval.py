import math

# ramanujan approximation of perimeter of oval
def approx(a, b):
    return math.pi * (3 * (a+b) - ((3*a + b) * (a + 3*b)) ** (1/2))

# perimeter: (1, 1-sqrt(3)/2) ~ 4.1
# will this work with the factor r?
