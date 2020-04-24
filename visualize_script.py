import visualize as viz
import os
import json


annotation_path = '../data/hw02_preds'
savepath = '../data/annotated_images/'

viz.visualize_all_images_with_bounding_boxes(prediction_path=os.path.join(annotation_path, 'preds_train.json'),
                                             save_path=savepath)
