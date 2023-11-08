import matplotlib.pyplot as plt
import numpy as np
import math

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
################################################################ methods #######################################################################

def gaussian_method(matrix, right_side):
    system = concat_right_side_to_matrix(matrix, right_side)    
    for i in range(0, len(system)):
        print(system)
        system = rotation_for_gauss(system, i)
        for j in range(i+1, len(system)):
            if(system[j][i] != 0):
                system = subtract_lines_divided(system, j, i, system[i][i]/system[j][i])

    print(system)
    sol = np.zeros(len(right_side))
    for i in range(len(system)-1, -1, -1):
        print(i)
        accum = system[i][len(system[i])-1]
        for j in range(i+1, len(system)):
            accum -= system[i][j]*sol[j]
        
        print(accum)
        system[i][len(system)] = accum

        print("i end = ", system[i][len(system)])
        print("i, i = ", system[i][i])
        sol[i] = system[i][len(system)]/system[i][i]
        print(sol)

    return sol
    
def LU_decomposition(matrix, right_side):
    L = np.eye(len(matrix))
    for i in range(0, len(matrix)):
        for j in range(i+1, len(matrix)):
            if(matrix[j][i] != 0):
                val = matrix[i][i]/matrix[j][i]

    print(L)
    print(matrix)
    print(np.dot(L, matrix))

################################################################ testing code #######################################################################
a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 8.0]])
b = np.array([[1.0], [1.0], [1.0]])

LU_decomposition(a, b)

A = np.array([[50.0, 70.0], [70.0, 101.0]])
print(A)
print(np.dot(A, A))
print(np.linalg.inv(A))
print(np.linalg.eig(np.dot(A, A)))
print(np.linalg.eig(np.dot(np.linalg.inv(A), np.linalg.inv(A))))

print(1.0/2.25e+04)