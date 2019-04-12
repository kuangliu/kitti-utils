import cv2
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from calibration import Calibration
from utils import draw_lidar_points


img = cv2.imread('./data/training/image_2/000000.png')
img_h, img_w, _ = img.shape

calib = Calibration('./data/training/calib/000000.txt')

# Lidar points
pc = np.fromfile('./data/training/velodyne/000000.bin', dtype=np.float32)
pc = pc.reshape(-1, 4)  # Nx4
pc = pc[:, :3]

# Filter points in camera FOV
x_ref = calib.project_velo_to_ref(pc)
x_rect = calib.project_ref_to_cam(x_ref)
x_img = calib.project_cam_to_img(x_rect)

ids = (x_img[:, 0] < img_w) & (x_img[:, 0] >= 0) & \
    (x_img[:, 1] < img_h) & (x_img[:, 1] >= 0)
clip_distance = 2.0
ids = ids & (pc[:, 0] > clip_distance)

pc = pc[ids, :]
x_rect = x_rect[ids, :]
x_img = x_img[ids, :]
draw_lidar_points(pc)

# Draw lidar points on image
cmap = plt.cm.get_cmap('hsv', 256)
cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255
for i in range(x_rect.shape[0]):
    depth = x_rect[i, 2]
    color = cmap[int(640.0/depth), :]
    cv2.circle(img, (int(np.round(x_img[i, 0])),
                     int(np.round(x_img[i, 1]))), 2, color=tuple(color), thickness=-1)
Image.fromarray(img).show()
