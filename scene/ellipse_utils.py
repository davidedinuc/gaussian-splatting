import os
from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
from tqdm import tqdm
import torch
from torch.multiprocessing import Pool
import cv2
from typing import NamedTuple, Optional, List, Tuple
from scipy.special import softmax
from utils.graphics_utils import getWorld2View2
from utils.camera_utils import CameraInfo

def interp(x, xp, fp):
    # Flatten the input arrays
    x_flat = x.reshape(-1, x.shape[-1])
    xp_flat = xp.reshape(-1, xp.shape[-1])
    fp_flat = fp.reshape(-1, fp.shape[-1])

    # Perform interpolation for each set of flattened arrays
    ret_flat = np.array([np.interp(xf, xpf, fpf) for xf, xpf, fpf in zip(x_flat, xp_flat, fp_flat)])

    # Reshape the result to match the input shape
    ret = ret_flat.reshape(x.shape)
    return ret


def sorted_interp(x, xp, fp):
    # Identify the location in `xp` that corresponds to each `x`.
    # The final `True` index in `mask` is the start of the matching interval.
    mask = x[..., None, :] >= xp[..., :, None]

    def find_interval(x):
        # Grab the value where `mask` switches from True to False, and vice versa.
        # This approach takes advantage of the fact that `x` is sorted.
        x0 = np.max(np.where(mask, x[..., None], x[..., :1, None]), -2)
        x1 = np.min(np.where(~mask, x[..., None], x[..., -1:, None]), -2)
        return x0, x1

    fp0, fp1 = find_interval(fp)
    xp0, xp1 = find_interval(xp)
    with np.errstate(divide='ignore', invalid='ignore'):
        offset = np.clip(np.nan_to_num((x - xp0) / (xp1 - xp0), nan=0.0), 0, 1)
    ret = fp0 + offset * (fp1 - fp0)
    return ret

def integrate_weights(w):
    """Compute the cumulative sum of w, assuming all weight vectors sum to 1.

    The output's size on the last dimension is one greater than that of the input,
    because we're computing the integral corresponding to the endpoints of a step
    function, not the integral of the interior/bin values.

    Args:
        w: Tensor, which will be integrated along the last axis. This is assumed to
        sum to 1 along the last axis, and this function will (silently) break if
        that is not the case.

    Returns:
        cw0: Tensor, the integral of w, where cw0[..., 0] = 0 and cw0[..., -1] = 1
    """
    cw = np.minimum(1, np.cumsum(w[..., :-1], axis=-1))
    shape = cw.shape[:-1] + (1,)
    # Ensure that the CDF starts with exactly 0 and ends with exactly 1.
    cw0 = np.concatenate([np.zeros(shape), cw, np.ones(shape)], axis=-1)
    return cw0

def invert_cdf(u, t, w_logits, use_gpu_resampling=False):
    """Invert the CDF defined by (t, w) at the points specified by u in [0, 1)."""
    # Compute the PDF and CDF for each weight vector.
    w = softmax(w_logits, axis=-1)
    cw = integrate_weights(w)

    # Interpolate into the inverse CDF.
    interp_fn = interp if use_gpu_resampling else sorted_interp  # Assuming these are defined using NumPy
    t_new = interp_fn(u, cw, t)
    return t_new

def sample(rng,
           t,
           w_logits,
           num_samples,
           single_jitter=False,
           deterministic_center=False,
           use_gpu_resampling=False):
    """Piecewise-Constant PDF sampling from a step function.

    Args:
        rng: random number generator (or None for `linspace` sampling).
        t: [..., num_bins + 1], bin endpoint coordinates (must be sorted)
        w_logits: [..., num_bins], logits corresponding to bin weights
        num_samples: int, the number of samples.
        single_jitter: bool, if True, jitter every sample along each ray by the same
        amount in the inverse CDF. Otherwise, jitter each sample independently.
        deterministic_center: bool, if False, when `rng` is None return samples that
        linspace the entire PDF. If True, skip the front and back of the linspace
        so that the centers of each PDF interval are returned.
        use_gpu_resampling: bool, If True this resamples the rays based on a
        "gather" instruction, which is fast on GPUs but slow on TPUs. If False,
        this resamples the rays based on brute-force searches, which is fast on
        TPUs, but slow on GPUs.

    Returns:
        t_samples: jnp.ndarray(float32), [batch_size, num_samples].
    """
    eps = np.finfo(np.float32).eps

    # Draw uniform samples.
    if rng is None:
        # Match the behavior of jax.random.uniform() by spanning [0, 1-eps].
        if deterministic_center:
            pad = 1 / (2 * num_samples)
            u = np.linspace(pad, 1. - pad - eps, num_samples)
        else:
            u = np.linspace(0, 1. - eps, num_samples)
            u = np.broadcast_to(u, t.shape[:-1] + (num_samples,))
    else:
        # `u` is in [0, 1) --- it can be zero, but it can never be 1.
        u_max = eps + (1 - eps) / num_samples
        max_jitter = (1 - u_max) / (num_samples - 1) - eps
        d = 1 if single_jitter else num_samples
        u = (
            np.linspace(0, 1 - u_max, num_samples) +
            rng.uniform(size=t.shape[:-1] + (d,), high=max_jitter))

    return invert_cdf(u, t, w_logits, use_gpu_resampling=use_gpu_resampling)


def focus_point_fn(poses: np.ndarray) -> np.ndarray:
    """Calculate nearest point to all focal axes in poses."""
    directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
    m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
    mt_m = np.transpose(m, [0, 2, 1]) @ m
    focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
    return focus_pt

def normalize(x: np.ndarray) -> np.ndarray:
    """Normalization helper function."""
    return x / np.linalg.norm(x)

def viewmatrix(lookdir: np.ndarray, up: np.ndarray,
               position: np.ndarray) -> np.ndarray:
    """Construct lookat view matrix."""
    vec2 = normalize(lookdir)
    vec0 = normalize(np.cross(up, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, position], axis=1)
    return m

def generate_ellipse_path_from_poses(poses: np.ndarray,
                          n_frames: int = 120,
                          const_speed: bool = True,
                          z_variation: float = 0.,
                          z_phase: float = 0.) -> np.ndarray:
    """Generate an elliptical render path based on the given poses."""
    # Calculate the focal point for the path (cameras point toward this).
    center = focus_point_fn(poses)
    # Path height sits at z=0 (in middle of zero-mean capture pattern).
    offset = np.array([center[0], center[1], 0])

    # Calculate scaling for ellipse axes based on input camera positions.
    sc = np.percentile(np.abs(poses[:, :3, 3] - offset), 100, axis=0)
    # Use ellipse that is symmetric about the focal point in xy.
    low = -sc + offset
    high = sc + offset
    # Optional height variation need not be symmetric
    z_low = np.percentile((poses[:, :3, 3]), 0, axis=0)
    z_high = np.percentile((poses[:, :3, 3]), 100, axis=0)

    def get_positions(theta):
        # Interpolate between bounds with trig functions to get ellipse in x-y.
        # Optionally also interpolate in z to change camera height along path.
        return np.stack([
            low[0] + (high - low)[0] * (np.cos(theta) * .5 + .5),
            low[1] + (high - low)[1] * (np.sin(theta) * .5 + .5),
            z_variation * (z_low[2] + (z_high - z_low)[2] *
                        (np.cos(theta + 2 * np.pi * z_phase) * .5 + .5)),
        ], -1)

    theta = np.linspace(0, 2. * np.pi, n_frames + 1, endpoint=True)
    positions = get_positions(theta)
    print('theta[0]', theta[0])

    if const_speed:
        # Resample theta angles so that the velocity is closer to constant.
        lengths = np.linalg.norm(positions[1:] - positions[:-1], axis=-1)
        theta = sample(None, theta, np.log(lengths), n_frames + 1)
        positions = get_positions(theta)

    # Throw away duplicated last position.
    positions = positions[:-1]

    # Set path's up vector to axis closest to average of input pose up vectors.
    avg_up = poses[:, :3, 1].mean(0)
    avg_up = avg_up / np.linalg.norm(avg_up)
    ind_up = np.argmax(np.abs(avg_up))
    up = np.eye(3)[ind_up] * np.sign(avg_up[ind_up])

    return np.stack([viewmatrix(p - center, up, p) for p in positions])

def pad_poses(p: np.ndarray) -> np.ndarray:
    """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
    bottom = np.broadcast_to([0, 0, 0, 1.], p[..., :1, :4].shape)
    return np.concatenate([p[..., :3, :4], bottom], axis=-2)

def unpad_poses(p: np.ndarray) -> np.ndarray:
    """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
    return p[..., :3, :4]

def transform_poses_pca(poses: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Transforms poses so principal components lie on XYZ axes.

    Args:
        poses: a (N, 3, 4) array containing the cameras' camera to world transforms.

    Returns:
        A tuple (poses, transform), with the transformed poses and the applied
        camera_to_world transforms.
    """
    t = poses[:, :3, 3]
    t_mean = t.mean(axis=0)
    t = t - t_mean

    eigval, eigvec = np.linalg.eig(t.T @ t)
    # Sort eigenvectors in order of largest to smallest eigenvalue.
    inds = np.argsort(eigval)[::-1]
    eigvec = eigvec[:, inds]
    rot = eigvec.T
    if np.linalg.det(rot) < 0:
        rot = np.diag(np.array([1, 1, -1])) @ rot

    transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)
    poses_recentered = unpad_poses(transform @ pad_poses(poses))
    transform = np.concatenate([transform, np.eye(4)[3:]], axis=0)

    # Flip coordinate system if z component of y-axis is negative
    if poses_recentered.mean(axis=0)[2, 1] < 0:
        poses_recentered = np.diag(np.array([1, -1, -1])) @ poses_recentered
        transform = np.diag(np.array([1, -1, -1, 1])) @ transform

    # Just make sure it's it in the [-1, 1]^3 cube
    scale_factor = 1. / np.max(np.abs(poses_recentered[:, :3, 3]))
    poses_recentered[:, :3, 3] *= scale_factor
    print('scale_factor', scale_factor)
    return poses_recentered, transform, scale_factor

def invert_transform_poses_pca(poses_recentered, transform, scale_factor):
    poses_recentered[:, :3, 3] /= scale_factor
    transform_inv = np.linalg.inv(transform)
    poses_original = unpad_poses(transform_inv @ pad_poses(poses_recentered))
    return poses_original

def generate_ellipse_path_from_camera_infos(
        cam_infos: List[CameraInfo],
        n_frames: int = 120,
        const_speed: bool = False,
        z_variation: float = 0.,
        z_phase: float = 0.
    ) -> List[CameraInfo]:
    print(f'Generating ellipse path from {len(cam_infos)} camera infos ...')
    poses = np.array([np.linalg.inv(getWorld2View2(cam_info.R, cam_info.T))[:3, :4] for cam_info in cam_infos])
    poses[:, :, 1:3] *= -1
    poses, transform, scale_factor = transform_poses_pca(poses)
    render_poses = generate_ellipse_path_from_poses(poses, n_frames, const_speed, z_variation, z_phase)
    render_poses = invert_transform_poses_pca(render_poses, transform, scale_factor)
    render_poses[:, :, 1:3] *= -1
    ret_cam_infos = []
    for uid, pose in enumerate(render_poses):
        R = pose[:3, :3]
        c2w = np.eye(4)
        c2w[:3, :4] = pose
        T = np.linalg.inv(c2w)[:3, 3]
        cam_info = CameraInfo(
            uid = uid,
            R = R,
            T = T,
            FovY = cam_infos[0].FovY,
            FovX = cam_infos[0].FovX,
            image = np.zeros_like(cam_infos[0].image),
            image_path = '',
            image_name = f'{uid:03d}.png',
            width = cam_infos[0].width,
            height = cam_infos[0].height,
            mask_path = '',
        )
        ret_cam_infos.append(cam_info)
    return ret_cam_infos


