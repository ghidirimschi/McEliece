import numpy as np
from constraint import *

H = np.array([[1, 0, 1, 0, 1, 0, 1],
              [0, 1, 1, 0, 0, 1, 1],
              [0, 0, 0, 1, 1, 1, 1]])

G = np.array([[1, 1, 0, 1],
              [1, 0, 1, 1],
              [1, 0, 0, 0],
              [0, 1, 1, 1],
              [0, 1, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])

R = np.array([[0, 0, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 0, 1]])


def random_binary_non_singular_matrix(n):
    a = np.random.randint(0, 2, size=(n, n))
    while np.linalg.det(a) == 0:
        a = np.random.randint(0, 2, size=(n, n))
    return a


S = random_binary_non_singular_matrix(4)
S_inv = np.linalg.inv(S).astype(int)


def generate_permutation_matrix(n):
    i = np.eye(n)
    p = np.random.permutation(i)
    return p.astype(int)


P = generate_permutation_matrix(7)
P_inv = np.linalg.inv(P).astype(int)

G_hat = np.transpose(np.mod((S.dot(np.transpose(G))).dot(P), 2))


def CSP_solve_matrix(G_hat, m, enc):
    problem = Problem()
    problem.addVariables(['a', 'b', 'c', 'd'], range(0, 2))

    def constraint_func(a, b, c, d):
        nparr = np.array([a, b, c, d])
        return np.array_equal(enc, np.mod(G_hat.dot(nparr),2))

    problem.addConstraint(constraint_func,
                              ['a', 'b', 'c', 'd'])

    sols = problem.getSolutions()
    idx = 0
    for sol in sols:
        print(f'Sol number {idx}')
        idx += 1
        print(sol)
        vec = [sol['a'], sol['b'], sol['c'], sol['d']]
        vec = np.array(vec)
        print(vec)
        print("Result verification of solution is")
        print(np.mod(G_hat.dot(vec), 2))
        print(enc)


print("G_hat is = \n", G_hat)
m = np.random.randint(2, size=4)
print("random message is = \n", m)
enc = np.mod(G_hat.dot(m), 2)
print('Encoded message is = ')
print(enc)
CSP_solve_matrix(G_hat, m, enc)
