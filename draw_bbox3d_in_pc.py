import cv2
import numpy as np

from calibration import Calibration
from utils import draw_lidar_points
from label_parser import KittiLabelParser


fname = '000010'
calib = Calibration('./data/training/calib/%s.txt' % fname)
label_parser = KittiLabelParser(root='./data/')

bboxes = []
for item in label_parser.parse_file(fname):
    if item['cls'] == 'DontCare':
        continue
    w, h, l = item['w'], item['h'], item['l']
    t = item['t']
    ry = item['ry']

    # obj coord.
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]

    x = np.vstack([x_corners, y_corners, z_corners])  # 3x8

    # obj -> camera
    s = np.sin(ry)
    c = np.cos(ry)
    R_obj_to_cam = np.array([[c, 0, s],
                             [0, 1, 0],
                             [-s, 0, c]])
    x = np.dot(R_obj_to_cam, x)  # 3x8
    x[0, :] = x[0, :] + t[0]
    x[1, :] = x[1, :] + t[1]
    x[2, :] = x[2, :] + t[2]

    x = x.transpose()  # 8x3

    # camera -> ref -> velodyne
    x = calib.project_cam_to_ref(x)
    x = calib.project_ref_to_velo(x)
    bboxes.append(x)

 # Lidar points
pc = np.fromfile('./data/training/velodyne/%s.bin' % fname, dtype=np.float32)
pc = pc.reshape(-1, 4)  # Nx4
pc = pc[:, :3]

draw_lidar_points(pc, bboxes)
