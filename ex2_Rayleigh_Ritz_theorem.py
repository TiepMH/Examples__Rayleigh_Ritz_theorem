import numpy as np


'''
Toy example 2:
This example is the extended version of the toy example 1
Let f(w) = (w^H A w) / (w^H B w) with w^H being Hermitian of w
Suppose we have the constraint ||w||^2 = 1
Using Cholesky decomposition, we can express B as B = C C^H
Defining D = (C^-1) A (C^H)^-1), we can rewrite f(w) as
    g(y) = (y^H W y) / (y^H y) where y = C^H w.
Note that
            g(y) = f(w)
Using the Rayleigh-Ritz theorem, we calculate
    y_opt = largest_eigenvector( D ),
    max g(y) = largest_eigenvalue( D )
Then we have
    max f(w) = max g(y),
    w_opt = (C^H)^-1 y_opt
In parallel, we randomly generate w_random
If the theorem is correct, then f(w_random) < f_opt
If the theorem is wrong, then count the number of violations
'''


def fw(w, A, B):
    wH = w.conj().T
    f = (wH@A@w)/(wH@B@w)
    return f


def gy(y, D):
    yH = y.conj().T
    f = (yH@D@y)/(yH@y)
    return f


A = np.random.randn(3, 3) + 1j*np.random.randn(3, 3)
A = A @ (A.conj().T)  # create a Hermitian matrix

B = np.random.randn(3, 3) + 1j*np.random.randn(3, 3)
B = B @ (B.conj().T)  # create another Hermitian matrix

w = np.random.randn(3, 1) + 1j*np.random.randn(3, 1)

f_w = fw(w, A, B)

# transformation
C = np.linalg.cholesky(B)  # Cholesky decomposition
CH = C.conj().T
C_inv = np.linalg.inv(C)  # inverse of C. NOTE: never use C**(-1)
CH_inv = np.linalg.inv(CH)  # inverse of C^H
y = CH @ w
D = C_inv @ A @ CH_inv  # transformation
g_y = gy(y, D)

print('f_w =', f_w)
print('g_y =', g_y)
print('We confirmed that f(w) = g(y).')
print('====================================')

""" We find the largest eigenvalue """
[eig_vals, eig_vecs] = np.linalg.eig(D)
index_max = np.argmax(eig_vals)
eig_val_max = eig_vals[index_max]
eig_vec_max = eig_vecs[:, index_max]

y_opt = eig_vec_max
w_opt = CH_inv @ y_opt

g_y_opt = gy(y_opt, D)
f_w_opt = fw(w_opt, A, B)

print('g_y_opt =', np.round(g_y_opt, 2))
print('f_w_opt =', np.round(f_w_opt, 2))
print('eig_val_max =', np.round(eig_val_max, 2))
print('max_f(w) = max_g(y) = largest eigenvalue of D')
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
    of_random = fw(w_random, A, B)
    ''' NOTE:
    Use np.round(a/b, 6) > 1 to check if a is different from b
    Do not use the subtraction (a-b). Why?
    Because if a - b = 1e-12, the PC misunderstands that a is different from b
    But, it is just the computational error '''
    if np.round(of_random/eig_val_max, n_digits) > 1:
        n_violations += 1

print('n_violations = 0 means that the theorem is correct')
print('n_violations =', n_violations)
