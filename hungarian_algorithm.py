import copy
import numpy as np
from scipy.optimize import linear_sum_assignment


def calculate_IOU(box_1, box_2):
    c1, c2 = box_1['center'], box_2['center']
    h1, h2 = box_1['height'], box_2['height']
    w1, w2 = box_1['width'], box_2['width']

    w_i = min(c1[0] + w1, c2[0] + w2) - max(c1[0], c2[0])
    h_i = min(c1[1] + h1, c2[1] + h2) - max(c1[1], c2[1])
    if w_i <= 0 or h_i <= 0:  # No overlap
        return 0
    area_1 = w1 * h1
    area_2 = w2 * h2
    I = w_i * h_i
    U = area_1 + area_2 - I  # Union = Total Area - I
    return I / U


def filter_detections(assignment, row_amt, col_amt):
    new_assignment = []

    if row_amt > col_amt:  # unbalanced case 1

        col_indexes = set(range(col_amt))

        for coord in assignment:

            if coord[1] in col_indexes:
                new_assignment.append(coord)

    elif col_amt > row_amt:  # unbalanced case 2

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

    return assignment, marked_rows, marked_cols


def object_assign(curr_objs, prev_objs, frame_num, IOU_min):
    '''
    curr_objs - [[center, height, width, cls, prob]]
    prev_objs - [[ ids, center, height, width, cls, prob, last_frame_seen, active_track ], ...]
    '''

    if len(curr_objs) == 0:
        return prev_objs

    # STEP 1: CREATE COST MATRIX

    prev_exist_objs = [obj for obj in prev_objs if obj['active_track']]

    if len(prev_exist_objs) == 0:
        hidden_return = copy.deepcopy(prev_objs)
        last_id = prev_objs[-1]['id'] + 1

        for obj in curr_objs:
            new_obj = copy.deepcopy(obj)
            new_obj['id'] = last_id
            new_obj['last_frame_seen'] = frame_num
            hidden_return.append(new_obj)
            last_id += 1
            return hidden_return

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

    curr_unidentified_ids = np.where((cost_matrix == 1).all(axis=1))[0]
    prev_unidentified_ids = np.where((cost_matrix == 1).all(axis=0))[0]

    # reprocess cost matrix

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

    new_objs = copy.deepcopy(prev_exist_objs)
    next_new_id, curr_ids = prev_objs[-1]['id'] + 1, len(curr_objs)

    for row, col in zip(row_idx, col_idx):

        if row in curr_unidentified_ids or col in prev_unidentified_ids:
            obj = curr_objs[row]

            unidentified_object = dict()
            unidentified_object['id'] = next_new_id
            unidentified_object['center'] = obj['center']
            unidentified_object['height'] = obj['height']
            unidentified_object['width'] = obj['width']
            unidentified_object['class'] = obj['class']
            unidentified_object['prob'] = obj['prob']
            unidentified_object['last_frame_seen'] = frame_num
            unidentified_object['velocity'] = (0, 0)
            unidentified_object['active_track'] = True
            unidentified_object['cov'] = np.identity(6)

            next_new_id += 1
            new_objs.append(unidentified_object)

        else:
            new_objs[col]['height'] = curr_objs[row]['height']
            new_objs[col]['width'] = curr_objs[row]['width']
            new_objs[col]['class'] = curr_objs[row]['class']
            new_objs[col]['prob'] = curr_objs[row]['prob']
            new_objs[col]['cov'] = curr_objs[row]['cov']
            new_objs[col]['active_track'] = True
            new_objs[col]['velocity'] = curr_objs[row]['velocity']
            new_objs[col]['last_frame_seen'] = frame_num
            new_objs[col]['center'] = curr_objs[row]['center']

    if len(curr_objs) > len(prev_exist_objs):

        unidentified_ids = [x for x in range(curr_ids) if x not in row_idx]

        for id_val in unidentified_ids:
            obj = curr_objs[id_val]

            unidentified_object = dict()
            unidentified_object['id'] = next_new_id
            unidentified_object['center'] = obj['center']
            unidentified_object['height'] = obj['height']
            unidentified_object['width'] = obj['width']
            unidentified_object['class'] = obj['class']
            unidentified_object['prob'] = obj['prob']
            unidentified_object['last_frame_seen'] = frame_num
            unidentified_object['velocity'] = (0, 0)
            unidentified_object['active_track'] = True
            unidentified_object['cov'] = np.identity(6)

            new_objs.append(unidentified_object)

            next_new_id += 1

    final_objs, new_objs_cnt = [], 0
    
    for obj in prev_objs: 

        if obj['active_track']:
            final_objs.append(new_objs[new_objs_cnt])
            new_objs_cnt += 1

        else:
            final_objs.append(obj)

    while new_objs_cnt < len(new_objs):
        final_objs.append(new_objs[new_objs_cnt])
        new_objs_cnt += 1

    return final_objs
