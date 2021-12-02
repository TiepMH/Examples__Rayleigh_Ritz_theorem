import numpy as np


'''
Toy example 1:
PURPOSE: We confirm the correctness of RAYLEIGH-RITZ theorem.
METHOD: Let f(w) = w' M w / (w' w) with w' being Hermitian of w
        Using the theorem, we calculate w_opt.
        Then we calculate f(w_opt) = f_opt
        In parallel, we randomly generate w_random
        If the theorem is correct, then f(w_random) < f_opt
        If the theorem is wrong, then count the number of violations
'''

# create a positive semidefinite Hermitian matrix
A = np.random.randn(3, 3) + 1j*np.random.randn(3, 3)
M = (A.conj().T) @ A

[eig_vals, eig_vecs] = np.linalg.eig(M)
index_max = np.argmax(eig_vals) # find the POSITION of max eig_val
eig_val_max = eig_vals[index_max]
print('the maximum eigenvalue =', np.round(eig_val_max, 2))

w_opt = eig_vecs[:, index_max]
w_opt = w_opt/np.linalg.norm(w_opt)  # to make sure ||w||^2 = 1
w_opt = np.reshape(w_opt, [3, 1])  # reshape (1, 3) to (3, 1)
wH_opt = w_opt.conj().T
of_opt = (wH_opt @ M @ w_opt) / (wH_opt @ w_opt)
print('the maximum value of the ratio =', np.round(of_opt, 2))

n_violations = 0
n_digits = 6  # the number of digits after the decimal point
for k in range(1000):
    w_random = np.random.randn(3, 1)
    w_random = w_random/np.linalg.norm(w_random)
    wH_random = w_random.conj().T
    of_random = (wH_random @ M @ w_random) / (wH_random @ w_random)
    ''' NOTE:
    Use np.round(a/b, 6) > 1 to check if a is different from b
    Do not use the subtraction (a-b). Why?
    Because if a - b = 1e-12, the PC misunderstands that a is different from b
    But, it is just the computational error '''
    if np.round(of_random/of_opt, n_digits) > 1:
        n_violations += 1

print('n_violations = 0 means that the theorem is correct')
print('n_violations =', n_violations)
