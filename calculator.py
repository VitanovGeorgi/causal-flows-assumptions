
import math

a = 1 
b = -1
c = -1
d = 1
e = 2

m0, m1, m2, m3 = 0, 0, 1.5, 3

s0, s1, s2, s3 = 1, 2, 1, 3


cov_confounder = (e + c*d) * d * s0 ** 2 + a**2 * c * s1 **2 - a*c*d*m0 * m1
var_1 = a**2 * s1 ** 2 + d**2 * s0 ** 2
var_2 = (e + c*d) ** 2 * s0 ** 2 + a**2 * c**2 * s1**2 + b**2 * s2 ** 2

rho_confounder = cov_confounder / math.sqrt(var_1 * var_2)

print(rho_confounder)



rho = 0.7

cov_cov = a * c * s1  - rho * b * s2
var_cov = (a*c*s1)**2 + (b*s2)**2
rho_cov = cov_cov / math.sqrt(var_cov)

print(rho_cov)


