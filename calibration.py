import numpy as np


class Calibration:
    '''Calibration utils.

    1. vel_to_ref: rotate & translate point from velodyne coord. to reference camera(cam0) coord.
    2. ref_to_cam: rotate point from camera(cam0) coord. to main camera(cam2) coord.
    3. cam_to_img: rotate & translate point from main camera(cam2) coord. to image pixel coord.

    3d XYZ in <label>.txt are in rect camera coord.
    2d box xy are in image2 coord
    Points in <lidar>.bin are in Velodyne coord.

    y_image2 = P^2_rect * R0_rect * Tr_velo_to_cam * x_velo
             = P^2_rect * R0_rect * x_ref
             = P^2_rect * x_rect

    P^2_rect = [f^2_u,  0,      c^2_u,  -f^2_u b^2_x;
                0,      f^2_v,  c^2_v,  -f^2_v b^2_y;
                0,      0,      1,      0]

    image2 coord:
      ----> x-axis (u)
      |
      |
      v y-axis (v)

    velodyne coord:
      front x, left y, up z

    rect/ref camera coord:
      right x, down y, front z

    Reference:
       (KITTI paper): http://www.cvlibs.net/publications/Geiger2013IJRR.pdf
       https://github.com/barrykui/kitti_object_vis/blob/master/kitti_util.py
    '''

    def __init__(self, calib_filepath):
        calib = self.read_calib_file(calib_filepath)
        self.P_velo_to_ref = calib['Tr_velo_to_cam'].reshape(3, 4)  # 3x4
        self.R_ref_to_cam = calib['R0_rect'].reshape(3, 3)  # 3x3
        self.P_cam_to_img = calib['P2'].reshape(3, 4)  # 3x4

        self.P_ref_to_velo = self.inverse_transpose(self.P_velo_to_ref)  # 3x4

    def read_calib_file(self, calib_filepath):
        calib = {}
        with open(calib_filepath) as f:
            for line in f.readlines():
                line = line.strip()
                if len(line) == 0:
                    continue
                k, v = line.split(':')
                calib[k] = np.array([float(x) for x in v.split()])
        return calib

    def cart_to_homo(self, pts_3d):
        '''Convert Cartesian 3D points to Homogeneous 4D points.

        Args:
          pts_3d: nx3 points in Cartesian coord.

        Returns:
          nx4 points in Homogeneous coord.
        '''
        n = pts_3d.shape[0]
        return np.hstack([pts_3d, np.ones((n, 1))])

    def inverse_transpose(self, P):
        '''Inverse rigit body transform matrix (3x4).

        Args:
          P: 3x4 transpose matrix.

        Returns:
          3x4 inverse transpose matrix. 
        '''
        P2 = np.zeros_like(P)
        P2[:3, :3] = P[:3, :3].T
        P2[:3, 3] = np.dot(-P[:3, :3].T, P[:3, 3])
        return P2

    def project_velo_to_ref(self, pts_3d):
        '''Rotate & translate points from velodyne coord. to reference camera coord.

        Args:
          pts_3d: nx3 points in velodyne coord.

        Returns:
          nx3 points in reference camera coord.
        '''
        pts_3d = self.cart_to_homo(pts_3d)
        return np.dot(pts_3d, self.P_velo_to_ref.T)

    def project_ref_to_velo(self, pts_3d):
        '''Inverse velo_to_ref project.

        Args:
          pts_3d: nx3 points in reference camera coord.

        Returns:
          nx3 points in velodyne coord.
        '''
        pts_3d = self.cart_to_homo(pts_3d)
        return np.dot(pts_3d, self.P_ref_to_velo.T)

    def project_ref_to_cam(self, pts_3d):
        '''Rotate points from reference camera coord. to main camera coord.

        Args:
          pts_3d: nx3 points in reference camera coord.

        Returns:
          nx3 points in main camera coord.

        y = (R * x^T)^T
          = x * R^T
        '''
        return np.dot(pts_3d, self.R_ref_to_cam.T)

    def project_cam_to_ref(self, pts_3d):
        '''Inverse ref_to_cam project.

        Args:
          pts_3d: nx3 points in main camera coord.

        Returns:
          nx3 points in reference camera coord.

        y = (R * x^T)^T
          = x * R^T
        '''
        R_cam_to_ref = self.R_ref_to_cam.T
        return np.dot(pts_3d, R_cam_to_ref.T)

    def project_cam_to_img(self, pts_3d):
        '''Project points in main camera coord. to image/pixel coord.

        Args:
          pts_3d: nx3 points in main camera coord.

        Returns:
          nx2 points in image coord.
        '''
        pts_3d = self.cart_to_homo(pts_3d)
        pts_3d = np.dot(pts_3d, self.P_cam_to_img.T)  # nx3
        pts_3d[:, 0] /= pts_3d[:, 2]
        pts_3d[:, 1] /= pts_3d[:, 2]
        return pts_3d[:, :2]
