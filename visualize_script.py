import visualize as viz
import os
import json


viz.visualize_all_images_with_bounding_boxes()


gts_path = '../data/hw02_annotations'
viz.visualize_all_images_with_bounding_boxes(prediction_path=os.path.join(gts_path, 'annotations_train.json'))