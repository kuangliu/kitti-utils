import cv2
import numpy as np

from mayavi import mlab


def draw_projected_bbox3d_on_image(img, pts, color=(0, 255, 0), thickness=2):
    ''' Draw 3d bounding box in image
        pts: (8,3) array of vertices for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
    '''
    pts = pts.astype(np.int32)
    for k in range(0, 4):
        # Ref: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
        i, j = k, (k+1) % 4
        # use LINE_AA for opencv3
        #cv2.line(img, (pts[i,0],pts[i,1]), (pts[j,0],pts[j,1]), color, thickness, cv2.CV_AA)
        cv2.line(img, (pts[i, 0], pts[i, 1]),
                 (pts[j, 0], pts[j, 1]), color, thickness)
        i, j = k+4, (k+1) % 4 + 4
        cv2.line(img, (pts[i, 0], pts[i, 1]),
                 (pts[j, 0], pts[j, 1]), color, thickness)
        i, j = k, k+4
        cv2.line(img, (pts[i, 0], pts[i, 1]),
                 (pts[j, 0], pts[j, 1]), color, thickness)
    return img


def draw_lidar_points(pts):
    '''Draw lidar points.

    Args:
      pts: lidar points sized (n,3) of XYZ.
    '''
    fig = mlab.figure(figure=None, bgcolor=(0, 0, 0),
                      fgcolor=None, engine=None, size=(1600, 1000))
    color = pts[:, 2]
    mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], color, color=None,
                  mode='point', colormap='gnuplot', scale_factor=0.3, figure=fig)
    # draw origin
    mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='sphere', scale_factor=0.2)
    mlab.view(azimuth=180, elevation=70, focalpoint=[
              12.0909996, -1.04700089, -2.03249991], distance=62.0, figure=fig)
    mlab.show()
