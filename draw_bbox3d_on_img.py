import cv2
import numpy as np

from calibration import Calibration
from utils import draw_projected_bbox3d_on_image


img = cv2.imread('./data/training/image_2/000000.png')
calib = Calibration('./data/training/calib/000000.txt')

# Pedestrian 0.00 0 -0.20 712.40 143.00 810.73 307.92 1.89 0.48 1.20 1.84 1.47 8.41 0.01
xmin, ymin, xmax, ymax = 712.40, 143.00, 810.73, 307.92  # 2d bbox in image coord.
bbox2d = [xmin, ymin, xmax, ymax]
height, width, length = 1.89, 0.48, 1.20  # 3d object size
t = [1.84, 1.47, 8.41]  # location (x,y,z) in rectified camera coord.
ry = 0.01  # yaw angle in rectified camera coord. [-pi, pi]

# obj coord.
w, h, l = width, height, length
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

# camera -> img
x = calib.project_cam_to_img(x)  # 8x2
draw_projected_bbox3d_on_image(img, x)
cv2.imwrite('z.png', img)
