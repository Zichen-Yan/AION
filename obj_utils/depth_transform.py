import yaml
from typing import Dict, Any
import numpy as np
import cv2

class Config:
    def __init__(self,
                 sensor_cfg: Dict[str, Any] = None,
                 transform_cfg: Dict[str, Any] = None,
                 projection_cfg: Dict[str, Any] = None,
                 laserscan_cfg: Dict[str, Any] = None):
        # default
        self.sensor_cfg = sensor_cfg or {
            "fov_deg": [90.0, 90.0]
        }

        self.transform_cfg = transform_cfg or {
            "rotate_points": [['x', -30]],  # rotation
            "filter_points": [['y', -0.25, 0.25]],
        }

        self.projection_cfg = projection_cfg or {
            "map_resolution": 0.2,
            "map_size": 100,
        }
        self.laserscan_cfg = laserscan_cfg or {
            "n_intervals": 30,
            "default_value": 3,
        }

    @classmethod
    def from_yaml(cls, yaml_path: str):
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return cls(
            sensor_cfg=data.get("sensor_cfg", None),
            transform_cfg=data.get("transform_cfg", None),
            projection_cfg=data.get("projection_cfg", None),
            laserscan_cfg=data.get("laserscan_cfg", None),
        )


def depth_to_pointcloud(depth, fov_deg=[90.0, 90.0]):
    H, W = depth.shape
    fov_x, fov_y = np.deg2rad(fov_deg[0]), np.deg2rad(fov_deg[1])

    fx = W / (2 * np.tan(fov_x / 2))
    fy = H / (2 * np.tan(fov_y / 2))
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0

    Z = depth.astype(np.float32)

    u = np.arange(W)
    v = np.arange(H)
    uu, vv = np.meshgrid(u, v)

    X = (uu - cx) * Z / fx
    Y = -(vv - cy) * Z / fy

    pts = np.stack([X, Y, -Z], axis=-1)
    return pts


def rotate_points(points, rotates=None):
    if rotates is None or len(rotates) == 0:
        return points

    R = np.eye(3)

    for axis, theta_deg in rotates:
        theta = np.deg2rad(theta_deg)

        if isinstance(axis, str):
            axis = axis.lower()
            if axis == 'x':
                R_axis = np.array([
                    [1, 0, 0],
                    [0, np.cos(theta), -np.sin(theta)],
                    [0, np.sin(theta), np.cos(theta)]
                ])
            elif axis == 'y':
                R_axis = np.array([
                    [np.cos(theta), 0, np.sin(theta)],
                    [0, 1, 0],
                    [-np.sin(theta), 0, np.cos(theta)]
                ])
            elif axis == 'z':
                R_axis = np.array([
                    [np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]
                ])
            else:
                raise ValueError("axis must be 'x', 'y', 'z'")
        else:
            axis = np.asarray(axis, dtype=float)
            axis = axis / np.linalg.norm(axis)

            K = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ])
            R_axis = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
        R = R_axis @ R

    return points @ R.T


def filter_points(points, colors, filters=None):
    if filters is None or len(filters) == 0:
        return points, colors

    mask = np.ones(points.shape[0], dtype=bool)

    for item in filters:
        if isinstance(item, list) and len(item) == 3:
            axis, min_val, max_val = item
        else:
            raise
        if isinstance(axis, str):
            axis = axis.lower()
            axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis]
        else:
            axis_idx = axis

        if min_val is not None:
            mask &= points[:, axis_idx] >= min_val
        if max_val is not None:
            mask &= points[:, axis_idx] <= max_val

    return points[mask], colors[mask] if colors is not None else None


def depth_to_filted_pointcloud(depth,
                               rgb=None,
                               height=None,
                               cfg: Config = None):
    if cfg is None:
        cfg = Config()
    pts_ori = depth_to_pointcloud(depth, fov_deg=cfg.sensor_cfg['fov_deg'])
    pts = pts_ori.copy().reshape(-1, 3)  # (H*W, 3)
    mask = pts[:, 2] <= 0
    pts = pts[mask]

    color = None
    if rgb is not None:
        color = rgb.reshape(-1, 3)[mask] / 255.0

    if 'rotate_points' in cfg.transform_cfg:
        pts = rotate_points(pts, cfg.transform_cfg['rotate_points'])

    if 'filter_points' in cfg.transform_cfg:
        pts, color = filter_points(pts, color, cfg.transform_cfg['filter_points'])

    # Ground Filtering
    if height is not None:
        ground_filter = ['y', -height, None]
        pts, color = filter_points(pts, color, [ground_filter])

    return pts, color


def map_pts_to_intervals(pts,
                         fov_deg=[90.0, 90.0],
                         n_intervals=30,
                         default_value=3.0
                         ):
    angle = np.arctan2(np.abs(pts[:, 2]), pts[:, 0])  # 计算每个点的角度，弧度
    angle_boundaries = np.linspace(np.deg2rad(fov_deg[0] / 2), np.deg2rad(fov_deg[0] * 1.5), n_intervals + 1)
    angle_intervals, dist_intervals = [], []
    for i in range(n_intervals):
        start = angle_boundaries[i]
        end = angle_boundaries[i + 1]
        angle_intervals.append([start, end])
        if i == n_intervals - 1:
            mask = (angle >= start) & (angle <= end)
        else:
            mask = (angle >= start) & (angle < end)
        pts_in_interval = pts[mask]
        if len(pts_in_interval) > 0:
            dist_intervals.append(np.min(np.sqrt(pts_in_interval[:, 0] ** 2 + pts_in_interval[:, 2] ** 2)))
        else:
            dist_intervals.append(default_value)
    return np.array(angle_intervals), np.array(dist_intervals)


def depth_layer_scan(depth,
                     rgb=None,
                     height=None,
                     cfg: Config = None):
    pts, _ = depth_to_filted_pointcloud(depth,
                                        rgb=rgb,
                                        height=height,
                                        cfg=cfg)
    if pts.shape[0] == 0:
        return None, None, None, None

    angles_intervals, dist = map_pts_to_intervals(
        pts,
        fov_deg=cfg.sensor_cfg['fov_deg'],
        n_intervals=cfg.laserscan_cfg['n_intervals'],
        default_value=cfg.laserscan_cfg['default_value']
    )
    angles = (angles_intervals[:, 0] + angles_intervals[:, 1]) / 2.0
    dist[dist > cfg.laserscan_cfg['default_value']] = cfg.laserscan_cfg['default_value']  # restrict the max dis
    x_coord = dist * np.cos(angles)
    y_coord = dist * np.sin(angles)
    return x_coord, y_coord, angles, dist


def depth_layer_scan_api(depth,
                         rgb=None,
                         height=None,
                         fov_deg=[90.0, 90.0],
                         rotate_points=[['x', -30]],
                         filter_points=[['y', -0.25, 0.25]],
                         n_intervals=30,
                         default_value=3.0):
    """
    Project a depth image to a laser scan – API interface

    Parameters:
        depth: H×W depth image
        rgb: H×W×3 RGB image (optional)
        height: camera height, used for ground filtering (optional)
        n_intervals: number of angular intervals for depth scanning (default: 30)
        default_value: default distance value for empty intervals (default: 3)

    Returns:
        angles: laser scan angles, shape (n_intervals,), ndarray, in degrees
        dists: laser scan distances, shape (n_intervals,), ndarray
    """
    cfg = Config(
        sensor_cfg={"fov_deg": fov_deg},
        transform_cfg={"rotate_points": rotate_points, "filter_points": filter_points},
        laserscan_cfg={"n_intervals": n_intervals, "default_value": default_value}
    )
    _, _, angles, dist = depth_layer_scan(depth, rgb=rgb, height=height, cfg=cfg)
    if angles is not None and dist is not None:
        angles = np.rad2deg(angles)
        dist = dist / default_value
    else:
        angle_boundaries = np.linspace(fov_deg[0] / 2, fov_deg[0] * 1.5, n_intervals + 1)
        angles = (angle_boundaries[:-1] + angle_boundaries[1:]) / 2.0
        dist = np.ones_like(angles)
    return angles, dist

def rotate_points_x(points, theta_deg):
    """Rotate point cloud around the X-axis."""
    theta = np.deg2rad(theta_deg)
    R = np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])
    return points @ R.T

def _intrinsics_from_fov(width:int, height:int, fov_deg):
    fov_x, fov_y = np.deg2rad(fov_deg[0]), np.deg2rad(fov_deg[1])
    fx = width / (2.0 * np.tan(fov_x / 2.0))
    fy = height / (2.0 * np.tan(fov_y / 2.0))
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    return fx, fy, cx, cy

def pointcloud_to_depth(
    pts: np.ndarray,
    height: int,
    width: int,
    fov_deg=(90.0, 90.0),
    radius: int = 0,
    fill_value: float = 0.0,
) -> np.ndarray:
    """
    Project a point cloud in the camera frame to a pinhole depth image.

    Args:
        pts: (N,3) points in the SAME camera coordinate convention used by depth_to_pointcloud:
             - X right, Y up (OpenGL) or down (OpenCV), Z is forward-negative (pts[:,2] ≤ 0 for visible points).
             - Depth scalar is Z_pos = -pts[:,2] (> 0 for visible points).
        height, width: output image size (H, W)
        fov_deg: (fov_x_deg, fov_y_deg) to derive intrinsics (must match what was used to create pts)
        radius: integer splat radius in pixels; 0 = single-pixel write; >0 fills a (2r+1)x(2r+1) square.
        fill_value: value for pixels with no projected points.

    Returns:
        depth: (H,W) float32 array. Each pixel stores depth_in_meters.
    """
    assert pts.ndim == 2 and pts.shape[1] == 3, "pts must be of shape (N,3)"
    H, W = int(height), int(width)
    fx, fy, cx, cy = _intrinsics_from_fov(W, H, fov_deg)

    Z = -pts[:, 2]  # camera-forward depth must be positive
    valid = Z > 0.0
    if not np.any(valid):
        return np.full((H, W), fill_value, dtype=np.float32)

    X = pts[valid, 0]
    Y = pts[valid, 1]
    Zv = Z[valid]

    u = fx * (X / Zv) + cx
    v = -fy * (Y / Zv) + cy

    ui = np.round(u).astype(int)
    vi = np.round(v).astype(int)

    in_bounds = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    if not np.any(in_bounds):
        return np.full((H, W), fill_value, dtype=np.float32)

    ui = ui[in_bounds]
    vi = vi[in_bounds]
    Zv = Zv[in_bounds]

    depth = np.full((H, W), np.inf, dtype=np.float32)

    r = int(radius)
    for px, py, z in zip(ui, vi, Zv):
        x0 = max(0, px - r); x1 = min(W - 1, px + r)
        y0 = max(0, py - r); y1 = min(H - 1, py + r)
        block = depth[y0:y1+1, x0:x1+1]
        depth[y0:y1+1, x0:x1+1] = np.minimum(block, z)
    mask = np.isfinite(depth)
    out = np.full_like(depth, fill_value, dtype=np.float32)
    out[mask] = (depth[mask]).astype(np.float32)
    return out

def _process_depth_roi(depth_frame, percentage):
    """
    Robust version: detect far-region ROI using depth smoothing + Otsu threshold + connected components filtering.
    """
    h, w = depth_frame.shape
    abstract_info = {'center_x': 0.0, 'center_y': 0.0, 'mean_depth': np.nan, 'found_flag': False}

    valid_mask = (depth_frame<10.0) & (depth_frame > 2.0)
    if not np.any(valid_mask):
        return abstract_info

    depth_smooth = cv2.medianBlur(depth_frame.astype(np.float32), 5)
    depth_valid = depth_smooth[valid_mask]

    thr = np.quantile(depth_valid, percentage)
    mask = np.uint8((depth_smooth > thr) & valid_mask) * 255

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:  # 没有有效区域
        return abstract_info

    # Compute minimum area in pixels
    min_area_ratio = 0.01
    min_area = min_area_ratio * h * w

    # Find the largest valid component above the threshold
    valid_indices = [
        i for i in range(1, num_labels)
        if stats[i, cv2.CC_STAT_AREA] >= min_area
    ]
    if not valid_indices:
        return abstract_info

    largest_idx = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
    mask_clean = np.uint8(labels == largest_idx) * 255

    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        best = max(contours, key=cv2.contourArea)
        M = cv2.moments(best)
        if M["m00"] > 0:
            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            roi_depth_mean = np.nanmean(depth_frame[mask_clean > 0])

            x, y, bw, bh = cv2.boundingRect(best)
            abstract_info = {
                'center_x': (cx / w) - 0.5,
                'center_y': (cy / h) - 0.5,
                'W': bw / w,
                'H': bh / h,
                'area': cv2.contourArea(best) / (h * w),
                'mean_depth': float(roi_depth_mean),
                'found_flag': True,
            }

    return abstract_info

def get_depth_ROI(depth, fov=90, camera_pitch_degrees=30, height=None, percentage=0.65):
    """
    Process the depth map, with an option for perspective correction.

    Args:
        depth_frame: The original depth map.
        fov: Field of View in degrees.
        camera_pitch_degrees: The camera's pitch angle in degrees (negative means tilted down).
        correct_perspective: Whether to perform perspective correction.

    Returns:
        abstract_info: Bounding box information (in the original view's coordinate system).
        y_horizon: The position of the horizon line.
    """
    h, w = depth.shape

    cfg = Config(
        sensor_cfg={"fov_deg": [fov, fov]},
        transform_cfg={"rotate_points": [['x', -camera_pitch_degrees]], "filter_points": []},
    )
    pts_corrected, _ = depth_to_filted_pointcloud(depth, None, height, cfg)
    depth_corrected = pointcloud_to_depth(
                            pts_corrected, h, w,
                            fov_deg=[fov, fov],
                            radius=1,
                            fill_value=0.0)

    abstract_info_corrected = _process_depth_roi(depth_corrected, percentage=percentage)

    if abstract_info_corrected['found_flag']:
        # Map the bounding box center from the corrected view back to the original view
        cx = (abstract_info_corrected['center_x'] + 0.5) * w
        cy = (abstract_info_corrected['center_y'] + 0.5) * h

        # Create 3D coordinates for the center point
        center_depth = depth_corrected[int(cy), int(cx)] if depth_corrected[int(cy), int(cx)] > 0 else 2.0

        # Convert the center point to a 3D point
        fov_rad = np.deg2rad(fov)
        fx = w / (2 * np.tan(fov_rad / 2))
        fy = h / (2 * np.tan(fov_rad / 2))
        cx_cam, cy_cam = (w - 1) / 2.0, (h - 1) / 2.0

        X = (cx - cx_cam) * center_depth / fx
        Y = -(cy - cy_cam) * center_depth / fy  # OpenGL coordinate system
        Z = -center_depth

        # Rotate back to the original view
        center_3d = np.array([[X, Y, Z]])
        center_3d_orig = rotate_points_x(center_3d, camera_pitch_degrees)

        # Project back to image coordinates
        X_orig, Y_orig, Z_orig = center_3d_orig[0]
        if Z_orig < 0:  # Ensure the point is in front of the camera
            u_orig = int(-X_orig * fx / Z_orig + cx_cam)
            v_orig = int(Y_orig * fy / Z_orig + cy_cam)  # Note the sign

            # Update abstract_info to the coordinates in the original view
            abstract_info_corrected['center_x'] = (u_orig / w) - 0.5
            abstract_info_corrected['center_y'] = (v_orig / h) - 0.5

        abstract_info = abstract_info_corrected
    else:
        abstract_info = abstract_info_corrected

    # Calculate the horizon line for the original view
    vertical_fov_degrees = fov
    y_horizon = int(
        (h / 2.0) * (1.0 - np.tan(np.radians(camera_pitch_degrees)) /
                     np.tan(np.radians(vertical_fov_degrees / 2.0)))
    )
    y_horizon = max(0, min(h - 1, y_horizon))
    abstract_info['y_horizon'] = (y_horizon / h) - 0.5
    return abstract_info