import copy
import matplotlib.pyplot as plt
import numpy as np
from greens_function_nevp import surface_greens_function_poles


def group_velocity(eigenvector, eigenvalue, h_r):
    """
    Computes the group velocity of wave packets

    :param eigenvector:       eigenvector
    :type eigenvector:        numpy.matrix(dtype=numpy.complex)
    :param eigenvalue:        eigenvalue
    :type eigenvector:        numpy.complex
    :param h_r:               coupling Hamiltonian
    :type h_r:                numpy.matrix
    :return:                  group velocity for a pair consisting of an eigenvector and an eigenvalue
    """

    return np.imag(eigenvector.H * h_r * eigenvalue * eigenvector)


eps0 = 8.85e-12
eps_h = 1
c = 3.0e8
q = 1.6e-19
h = 1.054e-34

omega_p = 8.45 * q / h
l_p = 2 * np.pi / (eps_h * omega_p / c)
# 0.580 907
# omega = np.linspace(0.0033, 0.0034, 500) * omega_p
omega = np.linspace(0.52, 0.53, 100) * omega_p
k = eps_h * omega / c

d = l_p / 30
R = 0.25 * d
V = 4 * np.pi * R ** 3

# Drude model for particle


def inv_alpha(om):
    return 1.0 / (eps0 * V) * (1.0 / 3.0 - om ** 2 / omega_p ** 2)


def losses(om):
    k = eps_h * om / c
    return 1j * k ** 3 / (6 * np.pi * eps0)


# interparticle coupling

def A1(om, n):
    k = eps_h * om / c
    return np.exp(1j * k * np.abs(n * d)) / (4.0 * np.pi * eps0 * np.abs(n * d)) * (k ** 2)


def A2(om, n):
    k = eps_h * om / c
    return np.exp(1j * k * np.abs(n * d)) / (4.0 * np.pi * eps0 * np.abs(n * d)) * (
                1.0 / ((n * d) ** 2) - 1j * k / np.abs(n * d))


num_neighbours = 200

eigs = []
sgf_l = []
sgf_r = []

norm = 1e33

for E in omega:
    print(E)
    h = []
    for j in range(num_neighbours, 0, -1):
        h_l = np.matrix(np.zeros((1, 1), dtype=np.complex))
        h_l[0, 0] = A1(E, -j) - A2(E, -j)
        # h_l[1, 1] = A1(E, -j) - A2(E, -j)
        # h_l[0, 0] = 2 * A2(E, -j)
        h_l = h_l / norm
        h.append(h_l)

    h_0 = np.matrix(np.identity(1)) * (inv_alpha(E) + 0*losses(E))
    h_0 = h_0 / norm
    h.append(h_0)

    for j in range(1, num_neighbours+1):
        h_r = np.matrix(np.zeros((1, 1), dtype=np.complex))
        h_r[0, 0] = A1(E, j) - A2(E, j)
        # h_r[1, 1] = A1(E, j) - A2(E, j)
        # h_r[0, 0] = 2 * A2(E, j)

        h_r = h_r / norm
        h.append(h_r)

    vals, vects = surface_greens_function_poles(0, h)

    vals = np.diag(vals)
    # vals.flags['WRITEABLE'] = True
    #
    # for j, v in enumerate(vals):
    #     print "The element number is", j, vals[j]
    #     if np.abs(np.abs(v) - 1.0) > 0.03:
    #         vals[j] = float('nan')
    #     else:
    #         vals[j] = np.angle(v)
    #         print "The element number is", j, vals[j]

    eigs.append(vals)


plt.plot(omega, np.array(eigs), 'o')
plt.show()
print('hi')