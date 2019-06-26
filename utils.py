import cv2
import numpy as np

from mayavi import mlab


def draw_projected_box3d_on_image(img, pts, color=(0, 255, 0), thickness=2):
    ''' Draw 3d bounding box on image.
    
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


def draw_point_cloud(pc, boxes=None, colors=None):
    '''Draw draw point cloud.

    Args:
      pc: (np.array) point cloud, sized (n,3) of XYZ.
      boxes: (list(np.array)) list of 3D bounding boxes, each sized [8,3].
      colors: (list(tuple)) list of RGB colors.
    '''
    fig = mlab.figure(figure=None, bgcolor=(0, 0, 0),
                      fgcolor=None, engine=None, size=(1600, 1000))
    color = pc[:, 2]
    mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], color, color=None,
                  mode='point', colormap='gnuplot', scale_factor=0.3, figure=fig)

    # draw origin
    mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='sphere', scale_factor=0.2)
    mlab.view(azimuth=180, elevation=70, focalpoint=[
              12.0909996, -1.04700089, -2.03249991], distance=62.0, figure=fig)

    # draw 3d boxes
    if boxes is not None:
        if colors is None:
            colors = [(1, 1, 1)] * len(boxes)  # set default color to white

        for box, color in zip(boxes, colors):
            for k in range(0, 4):
                i, j = k, (k+1) % 4
                p, q = box[i], box[j]
                mlab.plot3d([p[0], q[0]], [p[1], q[1]],
                            [p[2], q[2]], color=color)
                i, j = k+4, (k+1) % 4 + 4
                p, q = box[i], box[j]
                mlab.plot3d([p[0], q[0]], [p[1], q[1]],
                            [p[2], q[2]], color=color)
                i, j = k, k+4
                p, q = box[i], box[j]
                mlab.plot3d([p[0], q[0]], [p[1], q[1]],
                            [p[2], q[2]], color=color)
    mlab.show()
