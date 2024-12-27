from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml")

metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])  # cfg.DATASETS.TRAIN = ["coco_2017_train", "coco_2017_val"]

print("\n -- 학습된 COCO DataSet 클래스 --")
for idx, class_name in enumerate(metadata.stuff_classes):
    print(f"{idx + 1}: {class_name}")