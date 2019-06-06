'''Point cloud augmentation.'''
import random
import numpy as np


def random_scale(pc, scale_range=[0.9, 1.1]):
    '''Randomly scale the point cloud.

    Args:
      pc: (np.array) point cloud, sized [N,4].
      scale_range: (list) scale range.

    Return:
      scaled point cloud, sized [N,4].
      scale param.

    Velodyne coord:
      front x, left y, up z.
    '''
    scale = random.uniform(*scale_range)
    pc = scale * pc
    return pc


def random_shift(pc, x_range=[0, 0], y_range=[0, 0], z_range=[0, 0]):
    '''Randomly shift the point cloud.

    Args:
      pc: (np.array) point cloud, sized [N,4].
      x_range: (list) x shift range.
      y_range: (list) y shift range.
      z_range: (list) z shift range.

    Return:
      shifted point cloud, sized [N,4].
      shift param.

    Velodyne coord:
      front x, left y, up z.
    '''
    x = random.uniform(*x_range)
    y = random.uniform(*y_range)
    z = random.uniform(*z_range)
    pc[:, :3] += [x, y, z]
    return pc, (x, y, z)


def random_rotate(pc, rotate_range=[0, 0]):
    '''Randomly rotate point cloud along Z-axis.

    Args:
      pc: (np.array) point cloud, sized [N,4].
      rotate_range: (list) rotation range in radians.

    Returns:
      rotated point cloud, sized [N,4].
      rotate param.

    Velodyne coord:
      front x, left y, up z.
    '''
    y = random.uniform(*rotate_range)
    ymtx = np.array([
        [np.cos(y), -np.sin(y), 0],
        [np.sin(y), np.cos(y), 0],
        [0, 0, 1]
    ])
    pc[:, :3] = pc[:, :3].dot(ymtx.T)
    return pc, ymtx


def flip(pc):
    pc[:, 1] = -pc[:, 1]
    return pc


def test():
    pc = np.random.randn(10, 4)
    print(pc)
    print(random_shift(pc, (0, 5), (0, 5), (0, 5))[0])


# test()
