import cv2
import numpy as np

from calibration import Calibration
from label_parser import KittiLabelParser
from utils import draw_projected_bbox3d_on_image


fname = '000000'
img = cv2.imread('./data/training/image_2/%s.png' % fname)
calib = Calibration('./data/training/calib/%s.txt' % fname)
label_parser = KittiLabelParser(root='./data/', classes=KittiLabelParser.kitti_classes[:-1])

for x in label_parser.parse_boxes_in_camera(fname):
    # camera -> img
    x = calib.project_cam_to_img(x)  # 8x2
    draw_projected_bbox3d_on_image(img, x)
cv2.imwrite('z.png', img)
