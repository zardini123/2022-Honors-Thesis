import sympy

# Sympy docs: Creating a custom function:
#   https://docs.sympy.org/latest/guides/custom-functions.html#creating-a-custom-function
# class BernsteinPolynomial(sympy.Function):
#     @classmethod
#     def eval(cls, degree, index, parameter):
#         # Args: n, k
#         return sympy.binomial(degree, index) \
#             * (parameter ** index) \
#             * ((1 - parameter) ** (degree - index))


def bernstein_polynomial(degree, index, parameter):
    # Binomial args: sympy.binomial(n, k)
    return sympy.binomial(degree, index) \
        * (parameter ** index) \
        * ((1 - parameter) ** (degree - index))
