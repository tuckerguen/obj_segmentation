from detectron2.data.datasets import register_coco_instances

register_coco_instances('baxter_coco_train1', {}, "../baxter_dataset/annotations/baxter_coco_train.json", "dataset/train")
register_coco_instances('baxter_coco_val1', {}, "../baxter_dataset/annotations/baxter_coco_val.json", "dataset/val")