import numpy as np


'''
Toy example 4:
This example is the extended version of the toy example 3
Let
    A = u^H u and B = v^H v
and
    f(w) = (1 + alpha * w^H A w) / (1 + alpha * w^H B w)
with w^H being Hermitian of w. Note that alpha is a constant.
Suppose we have the constraint ||w||^2 = 1.
Using the Rayleigh-Ritz theorem, we directly calculate
    w_opt = largest_eigenvector( D ),
where
    D = (B + (1/alpha)I)^(-1) @ (A + (1/alpha)I).
In short, we have
    max f(w) = f(w_opt) = largest eigenvalue of D,
    w_opt = largest eigenvector of D
'''


def fw(w, alpha, A, B):
    wH = w.conj().T
    f = (1 + alpha * wH@A@w)/(1 + alpha * wH@B@w)
    return f


alpha = 10

u = np.random.randn(3, 1) + 1j*np.random.randn(3, 1)
uH = u.conj().T
A = u @ uH  # create a Hermitian matrix

v = np.random.randn(3, 1) + 1j*np.random.randn(3, 1)
vH = v.conj().T
B = v @ vH  # create another Hermitian matrix

# calculate the matrix D
D = np.linalg.inv(B + (1/alpha)*np.eye(3)) @ (A + (1/alpha)*np.eye(3))

""" We find the largest eigenvalue """
[eig_vals, eig_vecs] = np.linalg.eig(D)
index_max = np.argmax(eig_vals)
eig_val_max = eig_vals[index_max]
eig_vec_max = eig_vecs[:, index_max]

""" Maximum value of f(w) """
w_opt = eig_vec_max.reshape([3, 1])
f_opt = fw(w_opt, alpha, A, B)

print('eig_val_max =', np.round(eig_val_max, 2))
print('f(eig_vec_max) =', np.round(f_opt, 2))
print('max_f(w) = f(w_opt) = the largest eigenvalue of D')
print('====================================')
print('w_opt = eigenvector corres. to the largest eigenvalue')
print('====================================')

""" We randomly generate w_random and calculate f(w_random).
    We then check if there is any violation f(w_random) > f_w_opt
"""
n_violations = 0
n_digits = 6  # the number of digits after the decimal point
for k in range(1000):
    w_random = np.random.randn(3, 1) + 1j*np.random.randn(3, 1)
    w_random = w_random/np.linalg.norm(w_random)
    wH_random = w_random.conj().T
    of_random = fw(w_random, alpha, A, B)
    ''' NOTE:
    Use np.round(a/b, 6) > 1 to check if a is different from b
    Do not use the subtraction (a-b). Why?
    Because if a - b = 1e-12, the PC misunderstands that a is different from b
    But, it is just the computational error '''
    if np.round(of_random/eig_val_max, n_digits) > 1:
        n_violations += 1

print('n_violations = 0 means that the theorem is correct')
print('n_violations =', n_violations)
