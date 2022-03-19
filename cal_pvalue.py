import numpy as np
from scipy.misc import comb
import math
from fractions import Fraction

""" calculate the p value according to 
https://en.wikipedia.org/wiki/McNemar%27s_test """


def cal_pvalue(c00, c11, c01, c10):
    """
    c00: int, argeed number on positive
    c01, c10: int, disagreed number
    c11: int, agreed number on negative
    return: p value
    """

    b = min(c01, c10)
    n = c01 + c10
    p = 0
    for i in range(b, n+1):
        cur_p = 1
        for j in range(i):
            cur_p *= ((n - j) / (2.0 * (i - j)))
        if not np.isinf(cur_p):# and n-i < 1000: 
            # print(n-i, cur_p)
            for i in range(n-i):
                cur_p /= 2.0
            p += cur_p
            # print(cur_p, p)
        # p = p + comb(c01+c10, j) / denominator
    # p /= (2 ** (c01 + c10 - 1))
    return 2 * p


def McNemar_test(c01, c10):
    """
    when c01 and c10 are big, chi square 
    https://en.wikipedia.org/wiki/McNemar%27s_test
    """
    
    chi_square = (c01 - c10) ** 2 / (c01 + c10)
    return chi_square


if __name__ == "__main__":
    c00 = 40099
    c11 = 57476
    c10 = 8956
    c01 = 9969
    print("McNemar chi square", McNemar_test(c01, c10))
    print("p value", cal_pvalue(c00, c11, c01, c10))