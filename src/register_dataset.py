import sys


def register(train_name, val_name):
    from detectron2.data.datasets import register_coco_instances

    register_coco_instances(train_name, {}, "../baxter_dataset/annotations/baxter_coco_train.json",
                            "dataset/train")
    register_coco_instances(val_name, {}, "../baxter_dataset/annotations/baxter_coco_val.json", "dataset/val")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Must pass in a training name and validation name to register baxter_coco dataset under')
        print('Usage:\npython register_dataset train_name val_name')
    else:
        register(sys.argv[1], sys.argv[2])
