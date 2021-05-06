import numpy as np
import rospkg
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, Metadata

setup_logger()

model = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
# model = 'LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml'


class ObjectSegmenter:

    def __init__(self, detection_thresh=0.3, use_default_weights=True):
        self.detection_thresh = detection_thresh
        # Configure predictor w/ COCO model
        self.cfg = self.get_model_cfg(use_default_weights=use_default_weights)
        self.predictor = DefaultPredictor(self.cfg)
        print('created segmenter')

    def get_classes(self, outputs):
        return outputs["instances"].pred_classes

    def get_boxes(self, outputs):
        return outputs["instances"].pred_boxes

    def get_model_cfg(self, use_default_weights=True):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(model))
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
        if use_default_weights:
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)  # Let training initialize from model zoo
        else:
            print('Using baxter_coco_train1 dataset')
            cfg.DATASETS.TRAIN = ('baxter_coco_train1',)
            MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes = ['sports ball', 'carrot', 'remote']
            print(cfg.DATASETS.TRAIN)
            path = rospkg.RosPack().get_path('obj_segmentation')
            cfg.MODEL.WEIGHTS = f'{path}/baxter_dataset/model_final.pth'
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.detection_thresh  # set threshold for this model
        return cfg

    def segment(self, data, y_crop_idx=0):
        im = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
        # Crop out bottom of image (baxter specific)
        if y_crop_idx <= im.shape[0]:
            im = im[:y_crop_idx, :]
        outputs = self.predictor(im)
        return outputs, im

    def get_labels(self, outputs):
        predictions = outputs["instances"]
        metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])
        classes = predictions.pred_classes if predictions.has("pred_classes") else None
        class_names = metadata.get("thing_classes", None)
        labels = [class_names[i] for i in classes]
        return labels

    def draw_output(self, outputs, im):
        # Get class labels found in image
        metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])
        # Draw segmentations on image
        v = Visualizer(im[:, :, ::-1], metadata, scale=1)
        return v.draw_instance_predictions(outputs["instances"].to("cpu"))




