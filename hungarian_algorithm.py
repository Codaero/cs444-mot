import copy
import numpy as np
from scipy.optimize import linear_sum_assignment


def calculate_IOU(box_1, box_2):
    c1, c2 = box_1[0], box_2[1]
    h1, h2 = box_1[1], box_2[2]
    w1, w2 = box_1[2], box_2[3]

    tl_1 = (c1[0] - w1 / 2, c1[1] - h1 / 2)
    br_1 = (c1[0] + w1 / 2, c1[1] + h1 / 2)
    tl_2 = (c2[0] - w2 / 2, c2[1] - h2 / 2)
    br_2 = (c2[0] + w2 / 2, c2[1] + h2 / 2)

    tl_iou = (max(tl_1[0], tl_2[0]), max(tl_1[1], tl_1[1]))
    br_iou = (max(br_1[0], br_2[0]), max(br_1[1], br_2[1]))

    intersection = abs(tl_iou[0] - br_iou[0]) * abs(tl_iou[1] - br_iou[1])
    area_1, area_2 = h1 * w1, h2 * w2
    union = area_1 + area_2 - intersection

    return intersection / union


def filter_detections(assignment, row_amt, col_amt):
    new_assignment = []

    if row_amt > col_amt: # unbalanced case 1

        col_indexes = set(range(col_amt))

        for coord in assignment:

            if coord[1] in col_indexes:
                new_assignment.append(coord)

    elif col_amt > row_amt: # unbalanced case 2

        row_indexes = set(range(row_amt))

        for coord in assignment:

            if coord[0] in row_indexes:
                new_assignment.append(coord)
                
    else:
        new_assignment = assignment

    return np.array(new_assignment)


def search_optimum(cost_matrix):
    assignment = []

    bool_matrix = (cost_matrix == 0)

    while np.count_nonzero(bool_matrix == True):

        solution = [float('inf'), -1]

        for row_num, row in enumerate(bool_matrix):

            amount_zeros = np.count_nonzero(row == True)

            if amount_zeros > 0 and solution[0] > amount_zeros:
                solution[0] = amount_zeros
                solution[1] = row_num

        col_idx = np.where(bool_matrix[solution[1]] == True)[0][0]
        row_idx = solution[1]

        assignment.append([row_idx, col_idx])

        bool_matrix[:, col_idx] = False
        bool_matrix[row_idx, :] = False

    return np.array(assignment)


def get_strides(cost_matrix):

    assignment = search_optimum(copy.deepcopy(cost_matrix))

    all_rows = set(range(cost_matrix.shape[0]))
    zero_rows = set(assignment[:,1])
    unmarked_rows = list(all_rows - zero_rows) 

    marked_cols, cond = [], True

    while cond:

        cond = False

        for row_num in unmarked_rows:

            zero_col = np.where(cost_matrix[row_num] == 0)[0]

            for col_num in zero_col:
                
                if col_num not in marked_cols:

                    marked_cols.append(col_num)

                    cond = True

        for coord in assignment:

            row_num, col_num = coord

            if row_num not in unmarked_rows and col_num in marked_cols:

                unmarked_rows.append(row_num)

                cond = True

    marked_rows = list(all_rows - set(unmarked_rows))
    return assignment, marked_cols, marked_rows


def object_assign(curr_objs, prev_objs, frame_num, IOU_min):
    '''
    curr_objs - [[center, height, width, cls, prob, active_track]]
    prev_objs - [[ ids, center, height, width, cls, prob, last_frame_seen, active_track ], ...]
    '''

    # STEP 1: CREATE COST MATRIX

    prev_exist_objs = [obj for obj in prev_objs if obj[7]]

    cost_matrix = np.zeros((len(curr_objs), len(prev_exist_objs)))

    for i, c_obj in enumerate(curr_objs):

        for j, p_obj in enumerate(prev_exist_objs):

            iou = calculate_IOU(c_obj, p_obj)

            if iou <= IOU_min:

                iou = 0

            cost_matrix[i][j] = 1 - iou

    # STEP 2: PERFORM ASSIGNMENT
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

    assignment = filter_detections(assignment, row_amt, col_amt)

    row_idx, col_idx = assignment[:, 0], assignment[:, 1]

    # STEP 3: PARSE ASSIGNMENT
    if len(curr_objs) > len(prev_exist_objs):

        new_objs = copy.deepcopy(prev_exist_objs)

        for row, col in zip(row_idx, col_idx):
            new_objs[col] = [new_objs[col][0], curr_objs[row][0], curr_objs[row][1], curr_objs[row][2],
                             curr_objs[row][3], curr_objs[row][4], frame_num, True]

        next_new_id, curr_ids = new_objs[-1][0] + 1, len(curr_objs)

        unidentified_ids = [x for x in range(curr_ids) if x not in row_idx]

        for id_val in unidentified_ids:
            obj = curr_objs[id_val]

            new_objs.append([next_new_id, obj[0], obj[1], obj[2], obj[3], obj[4], frame_num, True])

            next_new_id += 1

    else:
        new_objs = copy.deepcopy(prev_exist_objs)

        for row, col in zip(row_idx, col_idx):
            new_objs[col] = [new_objs[col][0], curr_objs[row][0], curr_objs[row][1], curr_objs[row][2],
                             curr_objs[row][3], curr_objs[row][4], frame_num, True]


    final_objs, new_objs_cnt = [], 0
    
    for obj in prev_objs: 

        if obj[7]:
            final_objs.append(new_objs[new_objs_cnt])
            new_objs_cnt += 1

        else:
            final_objs.append(obj)


    while new_objs_cnt < len(new_objs):
        final_objs.append(new_objs[new_objs_cnt])
        new_objs_cnt += 1

    return final_objs
