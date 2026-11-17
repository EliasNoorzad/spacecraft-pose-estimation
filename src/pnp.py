"""Estimate HST's camera-relative 6-DoF pose from 2D–3D keypoint matches."""

import cv2
import numpy as np


def solve_pnp(points_2d, points_3d, camera_matrix):
    """Return (rotation matrix, translation vector), or None if PnP fails.

    The 37 points must be in the same order. Image coordinates are in pixels;
    3D coordinates are in the HST model's coordinate system.
    """
    points_2d = np.asarray(points_2d, dtype=np.float64)
    points_3d = np.asarray(points_3d, dtype=np.float64)
    camera_matrix = np.asarray(camera_matrix, dtype=np.float64)

    if points_2d.shape != (37, 2) or points_3d.shape != (37, 3):
        raise ValueError("Expected 37 2D points and 37 corresponding 3D points")
    if camera_matrix.shape != (3, 3):
        raise ValueError("Camera intrinsic matrix must have shape (3, 3)")
    if not (
        np.isfinite(points_2d).all()
        and np.isfinite(points_3d).all()
        and np.isfinite(camera_matrix).all()
    ):
        raise ValueError("PnP inputs must contain finite values")

    success, rvec, tvec = cv2.solvePnP(
        objectPoints=np.ascontiguousarray(points_3d),
        imagePoints=np.ascontiguousarray(points_2d),
        cameraMatrix=camera_matrix,
        distCoeffs=None,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return None

    rotation, _ = cv2.Rodrigues(rvec)
    return rotation, tvec.reshape(3)
