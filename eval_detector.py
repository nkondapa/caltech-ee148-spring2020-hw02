import os
import json
import numpy as np
import matplotlib.pyplot as plt

def compute_iou(box_1, box_2):
    '''
    This function takes a pair of bounding boxes and returns intersection-over-
    union (IoU) of two bounding boxes.
    '''

    tlx1, tly1, brx1, bry1 = box_1
    tlx2, tly2, brx2, bry2 = box_2

    boxes = [box_1, box_2]

    # compute x_overlap
    min_tlx = int(np.argmin([boxes[0][0], boxes[1][0]]))
    max_tlx = 1 - min_tlx
    itlx = None
    ibrx = None
    if boxes[min_tlx][0] <= boxes[max_tlx][0] <= boxes[min_tlx][2]:
        itlx = boxes[max_tlx][0]
        min_brx = int(np.argmin([boxes[0][2], boxes[1][2]]))
        ibrx = boxes[min_brx][2]

    # compute y_overlap
    min_tly = int(np.argmin([boxes[0][1], boxes[1][1]]))
    max_tly = 1 - min_tly
    itly = None
    ibry = None
    if boxes[min_tly][1] <= boxes[max_tly][1] <= boxes[min_tly][3]:
        itly = boxes[max_tly][1]
        min_bry = int(np.argmin([boxes[0][3], boxes[1][3]]))
        ibry = boxes[min_bry][3]

    if itlx is None or itly is None:
        return 0

    ixlen = ibrx - itlx
    iylen = ibry - itly

    intersect_area = ixlen * iylen

    box1_area = (box_1[0] - box_1[2]) * (box_1[1] - box_1[3])
    box2_area = (box_2[0] - box_2[2]) * (box_2[1] - box_2[3])

    # print(itlx, itly, ibrx, ibry)
    # print(box1_area, box2_area, intersect_area)
    iou = intersect_area / (box1_area + box2_area - intersect_area)

    assert (iou >= 0) and (iou <= 1.0)

    return iou


def compute_counts(preds, gts, iou_thr=0.5, conf_thr=0.5):
    '''
    This function takes a pair of dictionaries (with our JSON format; see ex.)
    corresponding to predicted and ground truth bounding boxes for a collection
    of images and returns the number of true positives, false positives, and
    false negatives.
    <preds> is a dictionary containing predicted bounding boxes and confidence
    scores for a collection of images.
    <gts> is a dictionary containing ground truth bounding boxes for a
    collection of images.
    '''
    TP = 0
    FP = 0
    FN = 0

    '''
    BEGIN YOUR CODE
    '''
    #conf_thr = 0
    #iou_thr = 0
    for pred_file, pred in preds.items():
        # if pred_file != 'RL-288.jpg':
        #     continue
        gt = gts[pred_file]
        cFN_check = [1] * len(gt)
        if not pred:
            TP += 0
            FP += 0
            FN += sum(cFN_check)
            continue

        cP = np.sum(np.array(pred)[:, 4] >= conf_thr)
        available_points = set(range(len(pred)))

        if len(gt) != 0:
            iou_matrix = np.zeros(shape=(len(gt), len(pred)))
        else:
            iou_matrix = None
            cFP = cP

            TP += 0
            FP += cFP
            FN += 0
            continue

        # print(pred_file)
        #print(gt)
        for i in range(len(gt)):
            for j in range(len(pred)):
                if pred[j][4] < conf_thr:
                    continue
                 #print(pred[j][:4], '\n', gt[i])
                iou = compute_iou(pred[j][:4], gt[i])
                #print(iou)
                # print()
                if iou > iou_thr:
                    iou_matrix[i, j] = iou

        cTP = 0
        # print(available_points)
        while np.sum(iou_matrix) != 0:
            for i in range(iou_matrix.shape[0]):
                if np.sum(iou_matrix[i, :]) == 0:
                    continue
                else:
                    sel_ind = int(np.argmax(iou_matrix[i, :]))
                    iou_matrix[:, sel_ind] = 0
                    cTP += 1
                    cFN_check[i] = 0

        cFP = cP - cTP
        mask = np.array(cFN_check) == 1
        # print(np.array(gt)[mask])
        cFN = sum(cFN_check)

        TP += cTP
        FP += cFP
        FN += cFN
    '''
    END YOUR CODE
    '''

    return TP, FP, FN

# set a path for predictions and annotations:
pred_name = 'hw02_weakened_preds'
#pred_name = 'hw02_preds'
preds_path = '../data/' + pred_name
gts_path = '../data/hw02_annotations'
save_path = '../data/output_figures'

# load splits:
split_path = '../data/hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# Set this parameter to True when you're done with algorithm development:
done_tweaking = True

'''
Load training data.
'''
with open(os.path.join(preds_path,'preds_train.json'),'r') as f:
    preds_train = json.load(f)

with open(os.path.join(gts_path, 'annotations_train.json'),'r') as f:
    gts_train = json.load(f)

if done_tweaking:

    '''
    Load test data.
    '''

    with open(os.path.join(preds_path,'preds_test.json'),'r') as f:
        preds_test = json.load(f)

    with open(os.path.join(gts_path, 'annotations_test.json'),'r') as f:
        gts_test = json.load(f)


# For a fixed IoU threshold, vary the confidence thresholds.
# The code below gives an example on the training set for one IoU threshold.

# using (ascending) list of confidence scores as thresholds
pred_list = []
for fname in preds_train:
    # print(preds_train[fname])
    if not preds_train[fname]:
        continue
    pred_list.append(np.array(preds_train[fname], dtype=float)[:, 4])

confidence_thrs = np.sort(np.concatenate(pred_list))
reduced_confidence_thrs = np.unique(np.around(confidence_thrs, decimals=5))
tp_train = np.zeros(len(confidence_thrs))
fp_train = np.zeros(len(confidence_thrs))
fn_train = np.zeros(len(confidence_thrs))
tp_test = np.zeros(len(confidence_thrs))
fp_test = np.zeros(len(confidence_thrs))
fn_test = np.zeros(len(confidence_thrs))
iou_thr = 0.75
for i, conf_thr in enumerate(reduced_confidence_thrs):
    tp_train[i], fp_train[i], fn_train[i] = compute_counts(preds_train, gts_train, iou_thr=iou_thr, conf_thr=conf_thr)
    tp_test[i], fp_test[i], fn_test[i] = compute_counts(preds_test, gts_test, iou_thr=iou_thr, conf_thr=conf_thr)
    # tp_train[i], fp_train[i], fn_train[i] = compute_counts(preds_train, gts_train, iou_thr=0.5, conf_thr=conf_thr)
    # break
# Plot training set PR curves

precision_train = np.zeros(len(confidence_thrs))
recall_train = np.zeros(len(confidence_thrs))
precision_test = np.zeros(len(confidence_thrs))
recall_test = np.zeros(len(confidence_thrs))
for i in range(len(confidence_thrs)):
    if fn_train[i] == 0:
        print(i, confidence_thrs[i], tp_train[i], fp_train[i], fn_train[i])
    precision_train[i] = tp_train[i] / (tp_train[i] + fp_train[i])
    recall_train[i] = tp_train[i] / (tp_train[i] + fn_train[i])
    precision_test[i] = tp_test[i] / (tp_test[i] + fp_test[i])
    recall_test[i] = tp_test[i] / (tp_test[i] + fn_test[i])

# print(recall_train)
# print(precision_train)
plt.figure()
plt.plot(recall_train, precision_train, '-o')
plt.xlabel('recall')
plt.ylabel('precision')
# plt.title('Precision-Recall Train | IOU = ' + str(iou_thr))
# plt.savefig(os.path.join(save_path, pred_name + '_test_train_iou=' + str(iou_thr) + '.png'))

if done_tweaking:
    pass
    print('Code for plotting test set PR curves.')
    plt.plot(recall_test, precision_test, '-o')
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('Weakened Precision-Recall Curves | IoU = ' + str(iou_thr))
    plt.legend(['train', 'test'])
    # plt.ylim([0, 1])
    # plt.xlim([0, 0.4])
    plt.savefig(os.path.join(save_path, pred_name + '_test_train_iou=' + str(iou_thr) + '.png'))

