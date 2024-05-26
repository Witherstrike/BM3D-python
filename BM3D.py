import copy

import cv2
import numpy as np

BLOCK_SIZE = 3  # [x - BLOCK_SIZE // 2, x + (BLOCK_SIZE + 1) // 2)
SEARCH_SIZE = 13  # [x - SEARCH_SIZE // 2, x + (SEARCH_SIZE + 1) // 2) is the range to find patches

SIGMA = 25
MATCH_THRESHOLD = SIGMA * 2.0
FILTER_3D_THRESHOLD = SIGMA * 2.7

DISTANCE_THRESHOLD = 2500  # Used in step 1 grouping
PATCHES_SIZE = 10  # Number of patches in a stack

WIEN_THRESHOLD = 400  # Used in step 2 grouping


def block_distance(block_a, block_b):
    if SIGMA > 40:
        block_a[np.abs(block_a) < MATCH_THRESHOLD] = 0
        block_b[np.abs(block_b) < MATCH_THRESHOLD] = 0
    return np.linalg.norm(block_a - block_b) ** 2 / (BLOCK_SIZE ** 2)

def BM3D(image):
    (cols, rows) = image.shape

    center_lb = 0 + BLOCK_SIZE // 2  # block center lower bound
    center_ub_col = cols - (BLOCK_SIZE + 1) // 2 + 1  # block center column upper bound
    center_ub_row = rows - (BLOCK_SIZE + 1) // 2 + 1  # block center row upper bound

    # Preprocessing, calculate DCTs of all blocks
    dct_blocks = np.zeros((cols, rows, BLOCK_SIZE, BLOCK_SIZE))
    for col in range(center_lb, center_ub_col):
        for row in range(center_lb, center_ub_row):
            block = image[col - BLOCK_SIZE // 2: col + (BLOCK_SIZE + 1) // 2,
                    row - BLOCK_SIZE // 2: row + (BLOCK_SIZE + 1) // 2]
            block = np.float32(block)
            dct_blocks[col, row, :, :] = cv2.dct(block)

    # ------------------------------------------ STEP 1 ------------------------------------------
    # Buffers for aggregation
    numerator_buffer = np.zeros((cols, rows))
    dominator_buffer = np.zeros((cols, rows))

    for col in range(center_lb, center_ub_col):
        for row in range(center_lb, center_ub_row):

            block_dct = dct_blocks[col, row]
            patches = []

            # Grouping: Search near blocks within range of SEARCH_SIZE
            for patch_col in range(max(center_lb, col - SEARCH_SIZE // 2),
                                   min(center_ub_col, col + (SEARCH_SIZE + 1) // 2)):
                for patch_row in range(max(center_lb, row - SEARCH_SIZE // 2),
                                       min(center_ub_row, row + (SEARCH_SIZE + 1) // 2)):
                    patch_dct = dct_blocks[patch_col, patch_row]
                    distance = block_distance(block_dct, patch_dct)

                    if distance < DISTANCE_THRESHOLD:
                        patches.append((distance, (patch_col, patch_row)))

            if len(patches) > PATCHES_SIZE:
                patches.sort(key=lambda x: x[0])
                patches = patches[:PATCHES_SIZE]

            # Collaborative Filtering: Apply the 3D transform
            block_3D = np.float32([dct_blocks[temp_col, temp_row] for (_, (temp_col, temp_row)) in patches])

            non_zero_cnt = 0
            for block_col in range(BLOCK_SIZE):
                for block_row in range(BLOCK_SIZE):
                    arr_dct = cv2.dct(block_3D[:, block_col, block_row])
                    arr_dct[np.abs(arr_dct) < FILTER_3D_THRESHOLD] = 0
                    non_zero_cnt += np.count_nonzero(arr_dct)
                    block_3D[:, block_col, block_row] = (cv2.idct(arr_dct)
                                                         .reshape(block_3D[:, block_col, block_row].shape))                                     

            # Aggregation: Update the buffers
            for i, (_, (patch_col, patch_row)) in enumerate(patches):
                patch = cv2.idct(block_3D[i])
                weight = 1 / non_zero_cnt if non_zero_cnt >= 1 else 1
                numerator_buffer[patch_col - BLOCK_SIZE // 2: patch_col + (BLOCK_SIZE + 1) // 2,
                                 patch_row - BLOCK_SIZE // 2: patch_row + (BLOCK_SIZE + 1) // 2] += patch * weight
                dominator_buffer[patch_col - BLOCK_SIZE // 2: patch_col + (BLOCK_SIZE + 1) // 2,
                                 patch_row - BLOCK_SIZE // 2: patch_row + (BLOCK_SIZE + 1) // 2] += weight

        print("Step1: ", col, "/", center_ub_col - 1)

    basic_estimate = copy.deepcopy(image)

    for col in range(cols):
        for row in range(rows):
            if dominator_buffer[col, row] != 0:
                basic_estimate[col, row] = numerator_buffer[col, row] / dominator_buffer[col, row]

    # ------------------------------------------ STEP 2 ------------------------------------------
    # Preprocessing, calculate DCTs of all blocks in the basic_estimate image
    dct_blocks_basic = np.zeros((cols, rows, BLOCK_SIZE, BLOCK_SIZE))
    for col in range(center_lb, center_ub_col):
        for row in range(center_lb, center_ub_row):
            block = basic_estimate[col - BLOCK_SIZE // 2: col + (BLOCK_SIZE + 1) // 2,
                                   row - BLOCK_SIZE // 2: row + (BLOCK_SIZE + 1) // 2]
            block = np.float32(block)
            dct_blocks_basic[col, row, :, :] = cv2.dct(block)

    # Buffers for aggregation
    numerator_buffer = np.zeros((cols, rows))
    dominator_buffer = np.zeros((cols, rows))

    for col in range(center_lb, center_ub_col):
        for row in range(center_lb, center_ub_row):

            block = basic_estimate[col - BLOCK_SIZE // 2: col + (BLOCK_SIZE + 1) // 2,
                    row - BLOCK_SIZE // 2: row + (BLOCK_SIZE + 1) // 2]
            patches = []

            # Grouping: Search near blocks within range of SEARCH_SIZE
            for patch_col in range(max(center_lb, col - SEARCH_SIZE // 2),
                                   min(center_ub_col, col + (SEARCH_SIZE + 1) // 2)):
                for patch_row in range(max(center_lb, row - SEARCH_SIZE // 2),
                                       min(center_ub_row, row + (SEARCH_SIZE + 1) // 2)):

                    patch = basic_estimate[patch_col - BLOCK_SIZE // 2: patch_col + (BLOCK_SIZE + 1) // 2,
                            patch_row - BLOCK_SIZE // 2: patch_row + (BLOCK_SIZE + 1) // 2]
                    distance = np.linalg.norm(block - patch) ** 2 / (BLOCK_SIZE ** 2)

                    if distance < WIEN_THRESHOLD:
                        patches.append((distance, (patch_col, patch_row)))

            if len(patches) > PATCHES_SIZE:
                patches.sort(key=lambda x: x[0])
                patches = patches[:PATCHES_SIZE]

            # Collaborative Filtering: Apply the 3D transform
            block_3D = np.float32([dct_blocks[temp_col, temp_row] for (_, (temp_col, temp_row)) in patches])
            block_3D_basic = np.float32([dct_blocks_basic[temp_col, temp_row] for (_, (temp_col, temp_row)) in patches])

            wien_weights = np.zeros((BLOCK_SIZE, BLOCK_SIZE))
            for block_col in range(BLOCK_SIZE):
                for block_row in range(BLOCK_SIZE):
                    arr_dct_basic = cv2.dct(block_3D_basic[:, block_col, block_row])
                    sq_norm = np.linalg.norm(arr_dct_basic) ** 2

                    wien_weight = sq_norm / (sq_norm + SIGMA ** 2)
                    wien_weights[block_col, block_row] = wien_weight

                    arr_dct = cv2.dct(block_3D[:, block_col, block_row])
                    arr_dct *= wien_weight
                    block_3D[:, block_col, block_row] = (cv2.idct(arr_dct)
                                                         .reshape(block_3D[:, block_col, block_row].shape))

            # Aggregation: Update the buffers
            for i, (_, (patch_col, patch_row)) in enumerate(patches):
                patch = cv2.idct(block_3D[i])
                numerator_buffer[patch_col - BLOCK_SIZE // 2: patch_col + (BLOCK_SIZE + 1) // 2,
                                 patch_row - BLOCK_SIZE // 2: patch_row + (BLOCK_SIZE + 1) // 2] += wien_weights * patch
                dominator_buffer[patch_col - BLOCK_SIZE // 2: patch_col + (BLOCK_SIZE + 1) // 2,
                                 patch_row - BLOCK_SIZE // 2: patch_row + (BLOCK_SIZE + 1) // 2] += wien_weights

        print("Step2: ", col, "/", center_ub_col - 1)

    final_estimate = copy.deepcopy(image)

    for col in range(cols):
        for row in range(rows):
            if dominator_buffer[col, row] != 0:
                final_estimate[col, row] = numerator_buffer[col, row] / dominator_buffer[col, row]

    return basic_estimate, final_estimate


image_in = cv2.imread("test2.jpg")
image_grey = cv2.cvtColor(image_in, cv2.COLOR_BGR2GRAY)

image_out1, image_out2 = BM3D(image_grey)
cv2.imwrite("out1.jpg", image_out1)
cv2.imwrite("out2.jpg", image_out2)
