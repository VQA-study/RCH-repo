import cv2
import numpy as np
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import torch


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # 임계값 0.5
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

predictor = DefaultPredictor(cfg)

image = cv2.imread("sample_image_table.png")

outputs = predictor(image)
segmentation = outputs["sem_seg"].argmax(dim=0).to("cpu").numpy()

metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

colored_seg = Visualizer(image[:, :, ::-1], metadata, scale=1.2)
result = colored_seg.draw_sem_seg(segmentation)

cv2.imshow('Source Image', image)
cv2.imshow('Segmentation Result', result.get_image()[:, :, ::-1])
cv2.waitKey(0)
cv2.destroyAllWindows()