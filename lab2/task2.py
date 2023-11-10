import matplotlib.pyplot as plt
import numpy as np
import math
import time

################################################################ helper functions #######################################################################

def concat_right_side_to_matrix(matrix, right_side):
    new_matrix = np.concatenate((matrix, right_side), axis=1)
    return new_matrix


def find_max_in_column_lower_nonzero (matrix, column_number):
    max = matrix[column_number][column_number]
    index = column_number
    
    for i in range(column_number, len(matrix)):
        if(matrix[i][column_number] > max and matrix[i][column_number] != 0):
            index = i
            max = matrix[i][column_number]
    
    if (max == 0):
        return len(matrix)+1

    return index

def swap_matrix_lines(matrix, first, second):
    tmp = np.copy(matrix[first])
    matrix[first] = matrix[second]
    matrix[second] = tmp
    return matrix

def rotation_for_gauss(matrix, column):
    ind = find_max_in_column_lower_nonzero(matrix, column)
    if(ind == len(matrix)+1):
        return
    return swap_matrix_lines(matrix, column, ind)

################################################################ matrix arithmetics #######################################################################

def divide_line_by_number(matrix, line, number):
    return np.divide(matrix[line], np.full((1, len(matrix[line])), number))
    
def subtract_lines(matrix, first, second):
    matrix[first] = np.add(matrix[first], np.multiply(matrix[second], np.full((1, len(matrix[second])), -1)))
    return matrix

def subtract_lines_divided(matrix, first, second, number):
    matrix[first] = np.add(matrix[first], np.multiply(matrix[second], np.full((1, len(matrix[second])), -1/number)))
    return matrix

def split_LDU(matrix):
    LDU = []
    L = np.zeros_like(matrix)
    D = np.zeros_like(matrix)
    U = np.zeros_like(matrix)

    for i in range(0, len(matrix)):
        D[i][i] = matrix[i][i]
    for i in range(0, len(matrix)):
        for j in range(i+1, len(matrix)):
            U[i][j] = matrix[i][j]

    for i in range(0, len(matrix)):
        for j in range(0, i):
            L[i][j] = matrix[i][j]

    LDU.append(L)
    LDU.append(D)
    LDU.append(U)
    return LDU

def discrepancy(matrix, right_side, u):
    Af = np.dot(matrix, u)
    discrepancy = np.zeros_like(right_side)
    for i in range(len(right_side)):
        discrepancy[i] = right_side[i]-Af[i]

    norm = 0.0
    for i in range(len(right_side)):
        norm += discrepancy[i]*discrepancy[i]

    return math.sqrt(norm)

def iteration_body(matrix, LplusU, Dinv_1, Dinv_2, u_current, right_side):
    i = 0
    arr = [[], []]
    discr = 1
    while (discr > 1e-10):
        u_current = -1*np.dot(np.dot(Dinv_1, LplusU), u_current) + np.dot(Dinv_2, right_side)
        i+=1
        discr = discrepancy(matrix, right_side, u_current)
        arr[0].append(i)
        arr[1].append(discr)
    
    plt.yscale("log")
    plt.plot(arr[0], arr[1], marker='o')
    plt.show()
    return u_current

def pivot_matrix(M):
    m = len(M)

    id_mat = np.eye(m)

    for j in range(m):
        row = max(range(j, m), key=lambda i: abs(M[i][j]))
        if j != row:
            id_mat[j], id_mat[row] = id_mat[row], id_mat[j]

    return id_mat

def lu_decomposition(A):
    n = len(A)

    L = np.zeros_like(A)
    U = np.zeros_like(A)

    P = pivot_matrix(A)
    PA = np.dot(P, A)

    for j in range(n):
        L[j][j] = 1.0

        for i in range(j+1):
            s1 = sum(U[k][j] * L[i][k] for k in range(i))
            U[i][j] = PA[i][j] - s1

        for i in range(j, n):
            s2 = sum(U[k][j] * L[i][k] for k in range(j))
            L[i][j] = (PA[i][j] - s2) / U[j][j]

    return (P, L, U)

################################################################ methods #######################################################################

def gaussian_method(matrix, right_side):
    system = concat_right_side_to_matrix(matrix, right_side)    
    for i in range(0, len(system)):
        system = rotation_for_gauss(system, i)
        for j in range(i+1, len(system)):
            if(system[j][i] != 0):
                system = subtract_lines_divided(system, j, i, system[i][i]/system[j][i])

    sol = np.zeros(len(right_side))
    for i in range(len(system)-1, -1, -1):
        accum = system[i][len(system[i])-1]
        for j in range(i+1, len(system)):
            accum -= system[i][j]*sol[j]
    
        system[i][len(system)] = accum

        sol[i] = system[i][len(system)]/system[i][i]

    return sol
    
def LU_solver(matrix, right_side):
    PLU = lu_decomposition(matrix)
    right_side = np.dot(PLU[0], right_side)

    system = concat_right_side_to_matrix(PLU[1], right_side)
    sol = np.zeros(len(right_side))
    for i in range(0, len(system)-1):
        accum = system[i][len(system[i])-1]
        for j in range(0, i):
            accum -= system[i][j]*sol[j]
    
        system[i][len(system)] = accum

        sol[i] = system[i][len(system)]/system[i][i]

    for i in range(len(sol)):
        right_side[i][0] = sol[i]

    system = concat_right_side_to_matrix(PLU[2], right_side)
    sol = np.zeros(len(right_side))
    for i in range(len(system)-1, -1, -1):
        accum = system[i][len(system[i])-1]
        for j in range(i+1, len(system)):
            accum -= system[i][j]*sol[j]
    
        system[i][len(system)] = accum

        sol[i] = system[i][len(system)]/system[i][i]

    return sol



def jacobi(matrix, right_side):
    u = np.zeros_like(right_side)
    LDU = split_LDU(matrix)
    
    Dinv = np.zeros_like(matrix)
    for i in range(len(matrix)):
        Dinv[i][i] = 1/LDU[1][i][i]

    LplusU = np.zeros_like(matrix)
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            LplusU[i][j] = LDU[0][i][j]

    for i in range(len(matrix)):
        for j in range(i+1, len(matrix)):
            LplusU[i][j] = LDU[2][i][j]

    solution = iteration_body(matrix, LplusU, Dinv, Dinv, u, right_side)
    return solution
    
def seidel(matrix, right_side, eps):
    n = len(matrix)
    u = np.zeros(n)  # zero vector
    arr = [[], []]
    
    counter = 0

    converge = False
    while not converge:
        u_new = np.copy(u)
        for i in range(n):
            s1 = sum(matrix[i][j] * u_new[j] for j in range(i))
            s2 = sum(matrix[i][j] * u[j] for j in range(i + 1, n))
            u_new[i] = (right_side[i] - s1 - s2) / matrix[i][i]

        converge = np.linalg.norm(u_new - u) <= eps
        u = u_new
        counter += 1
        arr[0].append(counter)
        arr[1].append(discrepancy(matrix, right_side, u))
    
    plt.yscale("log")
    plt.plot(arr[0], arr[1], marker='o')
    plt.show()
    return u

def relaxation(matrix, right_side, eps):
    u = np.zeros_like(right_side)
    LDU = split_LDU(matrix)
    omega = 1.2
    L = LDU[0]
    D = LDU[1]
    U = LDU[2]
    B   = (- np.linalg.inv(D + omega * L)).dot((omega - 1) * D + omega * U)
    F_b = omega * (np.linalg.inv(D + omega * L)).dot(right_side)

    arr = [[], []]
    k=0
    while (np.linalg.norm(right_side - matrix.dot(u), ord = 2) > eps):
        u = B.dot(u) + F_b
        k += 1
        arr[0].append(k)
        arr[1].append(np.linalg.norm(right_side - matrix.dot(u)))
    
    plt.yscale("log")
    plt.plot(arr[0], arr[1], marker='o')
    plt.show()
    
    return solution

    

################################################################ testing code #######################################################################

b_for_iters = np.zeros(100)
b = np.zeros((100, 1))
for i in range(0, 100):
    b_for_iters[i] = 100-i
    b[i][0] = 100-i

print(b_for_iters)
print(b)
a = np.zeros((100, 100))

for i in range(0, 100):
    a[0][i] = 1

for i in range(1, 99):
    for j in range(i-1, i+2):
        if(i!=j):
            a[i][j] = 1
        else:
            a[i][j] = 10

a[99][98] = 1
a[99][99] = 1

solution = gaussian_method(a, b)
print("Невязка для метода Гаусса с выбором главного элемента =", discrepancy(a, b, solution))
solution = LU_solver(a, b)
print("Невязка для LU_decomposition =", discrepancy(a, b, solution))

solution = jacobi(a, b_for_iters)

solution = seidel(a, b_for_iters, 1e-10)

solution = relaxation(a, b_for_iters, 1e-10)
