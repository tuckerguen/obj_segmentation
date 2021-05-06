import numpy as np
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

setup_logger()

# model = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
model = 'LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml'


class ObjectSegmenter:

    def __init__(self):
        # Configure predictor w/ COCO model
        self.cfg = get_cfg()
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
        self.predictor = DefaultPredictor(self.cfg)
        print('created segmenter')

    def get_classes(self, outputs):
        return outputs["instances"].pred_classes

    def get_boxes(self, outputs):
        return outputs["instances"].pred_boxes

    def get_model_cfg(self):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(model))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)
        return cfg

    def segment(self, data, y_crop_idx=0):
        im = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
        cfg = self.get_model_cfg()
        predictor = DefaultPredictor(cfg)
        # Crop out bottom of image (baxter specific)
        if y_crop_idx <= im.shape[0]:
            im = im[:y_crop_idx, :]
        outputs = predictor(im)
        return outputs, im

    def get_labels(self, outputs):
        predictions = outputs["instances"]
        cfg = self.get_model_cfg()
        metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        classes = predictions.pred_classes if predictions.has("pred_classes") else None
        class_names = metadata.get("thing_classes", None)
        labels = [class_names[i] for i in classes]
        return labels

    def draw_output(self, outputs, im):
        # Get class labels found in image
        cfg = self.get_model_cfg()
        metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        # Draw segmentations on image
        v = Visualizer(im[:, :, ::-1], metadata, scale=1)
        return v.draw_instance_predictions(outputs["instances"].to("cpu"))




