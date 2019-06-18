import cv2
import numpy as np

from calibration import Calibration
from utils import draw_lidar_points
from label_parser import KittiLabelParser


fname = '000010'
calib = Calibration('./data/training/calib/%s.txt' % fname)
label_parser = KittiLabelParser(root='./data/', classes=KittiLabelParser.kitti_classes[:-1])

bboxes = []
for x in label_parser.parse_boxes_in_camera(fname):
    x = calib.project_cam_to_ref(x)
    x = calib.project_ref_to_velo(x)
    bboxes.append(x)

 # Lidar points
pc = np.fromfile('./data/training/velodyne/%s.bin' % fname, dtype=np.float32)
pc = pc.reshape(-1, 4)  # Nx4
pc = pc[:, :3]

draw_lidar_points(pc, bboxes)
