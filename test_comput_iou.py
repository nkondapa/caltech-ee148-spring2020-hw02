import eval_detector as ed
import matplotlib.pyplot as plt

box1 = [100, 100, 200, 200]


# box2 = [125, 125, 175, 175]
# iou = ed.compute_iou(box1, box2)
# print(iou)
#
# box3 = [175, 125, 225, 175]
# iou = ed.compute_iou(box1, box3)
# print(iou)

box4 = [50, 125, 225, 175]
iou = ed.compute_iou(box1, box4)
print(iou)


box5 = [190, 100, 300, 200]
iou = ed.compute_iou(box1, box5)
print(iou)

# boxa = box1
# boxb = box4
# plt.scatter([boxa[0], boxa[2]], [[boxa[1], boxa[3]]], c='b')
# plt.scatter([boxb[0], boxb[2]], [[boxb[1], boxb[3]]], c='r')
# plt.show()