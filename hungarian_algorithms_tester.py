import numpy as np
from scipy.optimize import linear_sum_assignment
from hungarian_algorithm import get_strides
from hungarian_algorithm import filter_detections

# # Cost matrix
# cost_matrix = np.array([[0.2, 0.5, 0.7],
#                         [0.6, 0.3, 0.4],
#                         [0.8, 0.1, 0.9],
#                         [0.4, 0.6, 0.2]])

# # Find the optimal assignment using linear_sum_assignment
# row_ind, col_ind = linear_sum_assignment(cost_matrix)

# # Print the row and column indices of the optimal assignments
# print("Optimal Assignments:")
# for col, row in zip(col_ind, row_ind):
#     print(f"Current Object {row+1} -> Previous Object {col+1}")


# # Cost matrix
# cost_matrix = np.array([[0.2, 0.5, 0.7, 0.3],
#                         [0.6, 0.3, 0.4, 0.2],
#                         [0.8, 0.1, 0.9, 0.6]])


# # Find the optimal assignment using linear_sum_assignment
# row_ind, col_ind = linear_sum_assignment(cost_matrix)

# # Print the row and column indices of the optimal assignments
# print("Optimal Assignments:")
# for col, row in zip(col_ind, row_ind):
#     print(f"Current Object {row+1} -> Previous Object {col+1}")


# -----------------------------------------------------------------------------------------------------------

# Cost matrix
# cost_matrix = np.array([[0.2, 0.5, 0.7],
#                         [0.6, 0.3, 0.4],
#                         [0.8, 0.1, 0.9],
#                         [0.4, 0.6, 0.2]])

# cost_matrix = np.array([[0.2, 0.5, 0.7, 0.3],
#                         [0.6, 0.3, 0.4, 0.2],
#                         [0.8, 0.1, 0.9, 0.6]])

cost_matrix = np.array([[0.0, 0.0], [0.0, 0.0]])

# Add extra rows/columns
row_amt = cost_matrix.shape[0]
col_amt = cost_matrix.shape[1]

if row_amt > col_amt:
    extra_zeros = np.zeros((row_amt - col_amt, row_amt))
    cost_matrix = np.concatenate((cost_matrix, extra_zeros.T), axis=1)

elif col_amt > row_amt:
    extra_zeros = np.zeros((col_amt - row_amt, col_amt))
    cost_matrix = np.concatenate((cost_matrix, extra_zeros), axis=0)

# row subtract
min_row_values = np.min(cost_matrix, axis=1)
cost_matrix = (cost_matrix.T - min_row_values).T

# column subtract
min_col_values = np.min(cost_matrix, axis=0)
cost_matrix = cost_matrix - min_col_values


dim = cost_matrix.shape[0]

row_strides = []
col_strides = []

while len(row_strides) + len(col_strides) < dim:

    assignment, row_strides, col_strides = get_strides(cost_matrix)

    min_value = float('inf')

    if len(row_strides) + len(col_strides) < dim:

        # get minimum value in unmarked values
        for row_num, row in enumerate(cost_matrix):

            if row_num not in row_strides:

                for col_num, val in enumerate(row):

                    if col_num not in col_strides:

                        if min_value > cost_matrix[row_num][col_num]:

                            min_value = cost_matrix[row_num][col_num]

        # subtract minimum value to every unmarked values
        for row_num, row in enumerate(cost_matrix):

            if row_num not in row_strides:

                for col_num, val in enumerate(row):

                    if col_num not in col_strides:

                        cost_matrix[row_num][col_num] -= min_value

        # add minimum value to intersections
        for row_num in row_strides:

            for col_num in col_strides:

                cost_matrix[row_num][col_num] += min_value

print(assignment + 1)
assignment = filter_detections(assignment, row_amt, col_amt)
row_idx, col_idx = assignment[:,0], assignment[:,1]


# Print the row and column indices of the optimal assignments
print("Optimal Assignments:")
for col, row in zip(col_idx, row_idx):
    print(f"Current Object {row+1} -> Previous Object {col+1}")
