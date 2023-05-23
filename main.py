import numpy as np
from ortools.linear_solver import pywraplp
import time
import random
random.seed(42)
np.random.seed(42)

def exact_algorithm():
    start = time.time()

    # INPUT DATA
    file = open('input_3_int_variables.txt', 'r')
    amount_items = int(file.readline())
    size_backpack = int(file.readline())
    optimal_solution = int(file.readline())
    conflict_prob = 0.5
    print(optimal_solution)
    file.close()

    cost_weight_mat = np.loadtxt('input_test.txt')
    conflict_matrix = np.zeros((amount_items, amount_items))

    # FILLING CONFLICT MATRIX
    # random_array = np.random.uniform(0, 1, amount_items)
    # for i in range(amount_items):
    #     if(random_array[i] > conflict_prob):
    #         conflict_matrix[i][int(random_array[i]*amount_items-1)] = 1
    #         conflict_matrix[int(random_array[i]*amount_items-1)][i] = 1
    # for i in range(amount_items):
    #     conflict_matrix[i][i] = 0


    def get_solver():
        solver = pywraplp.Solver.CreateSolver('SCIP')
        if not solver:
            return None

        return solver

    solver = get_solver()

    # CREATE AND ADD VARIABLES
    x = {}  # - variables of items
    infinity = solver.infinity()
    for i in range(amount_items):
        x[i] = solver.IntVar(0, 1, 'x[%i]' % i)

    print('NumV = ', solver.NumVariables())

    # CREATE CONSTRAINTS

    constraint = solver.RowConstraint(0, size_backpack)
    for i in range(amount_items):
        constraint.SetCoefficient(x[i], cost_weight_mat[i][1])

    for i in range(amount_items):
        constraint = solver.RowConstraint(0, 1)
        for j in range(amount_items):
            constraint.SetCoefficient(x[j], conflict_matrix[i][j])


    print('Number of constraints =', solver.NumConstraints())

    # CREATE OBJECTIVE FUNC
    objective = solver.Objective()
    for i in range(amount_items):
        objective.SetCoefficient(x[i], cost_weight_mat[i][0])

    objective.SetMaximization()

    # PRINTING RESULTS
    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        print('Objective value =', solver.Objective().Value())
        for j in range(amount_items):
            print(x[j].name(), ' = ', x[j].solution_value())


    else:
        print('The problem does not have an optimal solution.')

    if (solver.Objective().Value() == optimal_solution):
        print("BALDEZH")
    else:
        print("PLOHO")
    end = time.time()
    print("total time: ", abs(end - start))

def weight_constraint(current_weight, max_weight, weight_to_add):
    if current_weight + weight_to_add < max_weight:
        return True
    else: return False

def athenian_constraints(confl_matrix, list_of_added_items, want_to_add_index):
    for i in list_of_added_items:
        if confl_matrix[int(want_to_add_index)][int(i)] == 1:
            return False
    return True





def greedy_algorithm():
    start = time.time()
    # INPUT DATA
    file = open('input_3_int_variables.txt', 'r')
    amount_items = int(file.readline())
    size_backpack = int(file.readline())
    optimal_solution = int(file.readline())

    #Псевдовероятность появления единичек
    conflict_prob = 0.4
    print(optimal_solution)
    file.close()

    cost_weight_mat = np.loadtxt('input_test.txt')
    conflict_matrix = np.zeros((amount_items, amount_items))

    # FILLING CONFLICT MATRIX
    # random_array = np.random.uniform(0, 1, amount_items)
    # for i in range(amount_items):
    #     if (random_array[i] > conflict_prob):
    #         conflict_matrix[i][int(random_array[i] * amount_items - 1)] = 1
    #         conflict_matrix[int(random_array[i] * amount_items - 1)][i] = 1
    # for i in range(amount_items):
    #     conflict_matrix[i][i] = 0
    # print(conflict_matrix)



    numeration = np.arange(amount_items)
    numeration = np.vstack(numeration)


    cost_weight_mat = np.append(cost_weight_mat, numeration, axis=1)

    cost_weight_mat_sorted = cost_weight_mat[cost_weight_mat[:,0]. argsort()[::-1]]


    current_weight = 0
    list_of_id_necessary_items = []
    founded_optimal_solution = 0



    for i in range(amount_items):
        if weight_constraint(current_weight, size_backpack, cost_weight_mat_sorted[i][1]) and athenian_constraints(conflict_matrix, list_of_id_necessary_items, cost_weight_mat_sorted[i][2]):
            founded_optimal_solution += cost_weight_mat_sorted[i][0]
            list_of_id_necessary_items.append(cost_weight_mat_sorted[i][2])
            current_weight += cost_weight_mat_sorted[i][1]
        else: continue
    print("Optimal solution: ", optimal_solution)
    print("Founded: ", founded_optimal_solution)
    print(list_of_id_necessary_items)
    end = time.time()
    print("Total time: ", end - start)

greedy_algorithm()