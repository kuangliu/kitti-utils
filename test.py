import numpy as np

from calibration import Calibration


calib = Calibration('./data/training/calib/000000.txt')

x = np.array([[1, 2, 3]])

y = calib.project_velo_to_ref(x)
z = calib.project_ref_to_velo(y)
print(z)

y = calib.project_ref_to_cam(x)
z = calib.project_cam_to_ref(y)
print(z)
