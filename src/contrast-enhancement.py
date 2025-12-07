"""
contrast_enhancement.py

Collection of contrast enhancement algorithms for grayscale images:

- HE                      : Global Histogram Equalization
- CLAHE                   : Classical CLAHE (OpenCV)
- BBHE                    : Brightness Preserving Bi-Histogram Equalization
- DHE (MPHE-style)        : Dynamic / Multi-Peak Histogram Equalization
- RMHE (WMSHE-style)      : Weighted / Range-Modified Histogram Equalization
- CLAHE_STATIC_JND        : CLAHE with a single JND-derived clip limit
- CLAHE_DYNAMIC_JND       : Tile-wise CLAHE with local JND-based clip limits

All functions:
    input  -> 2D NumPy array (grayscale image)
    output -> 2D NumPy uint8 array (enhanced image)
"""

from __future__ import annotations

from typing import Tuple
import numpy as np
import cv2


# -------------------------------------------------------------------------
# Helper utilities
# -------------------------------------------------------------------------

def _to_uint8(image: np.ndarray) -> np.ndarray:
    """
    Convert an image to uint8 [0, 255] if it is not already.
    """
    if image.dtype == np.uint8:
        return image

    img = image.astype(np.float32)
    img -= img.min()
    max_val = img.max()
    if max_val > 0:
        img /= max_val
    img = (img * 255.0).clip(0, 255)
    return img.astype(np.uint8)


def _check_grayscale(image: np.ndarray) -> None:
    """
    Raise an error if the image is not a 2D array.
    """
    if image.ndim != 2:
        raise ValueError("Expected a 2D grayscale image (H, W).")


def _find_peaks_1d(hist: np.ndarray, min_height: float) -> np.ndarray:
    """
    Simple local-maximum peak finder (to avoid SciPy dependency).

    Parameters
    ----------
    hist : np.ndarray
        1D histogram array.
    min_height : float
        Minimum height to accept a bin as a peak.

    Returns
    -------
    peaks : np.ndarray
        Indices of detected peaks.
    """
    peaks = []
    for i in range(1, len(hist) - 1):
        if hist[i] >= min_height and hist[i] >= hist[i - 1] and hist[i] >= hist[i + 1]:
            peaks.append(i)
    return np.array(peaks, dtype=int)



# 1) Global Histogram Equalization (HE)


def histogram_equalization(image: np.ndarray) -> np.ndarray:
    """
    Global Histogram Equalization (HE) using OpenCV.

    Parameters
    ----------
    image : np.ndarray
        2D grayscale image.

    Returns
    -------
    np.ndarray
        Equalized grayscale image (uint8).
    """
    _check_grayscale(image)
    img = _to_uint8(image)
    return cv2.equalizeHist(img)


# -------------------------------------------------------------------------
# 2) CLAHE (Classical)
# -------------------------------------------------------------------------

def clahe_opencv(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Classical CLAHE using OpenCV.

    Parameters
    ----------
    image : np.ndarray
        2D grayscale image.
    clip_limit : float
        Normalized clip limit (higher -> more contrast).
    tile_grid_size : (int, int)
        Size of the grid for the histogram equalization.

    Returns
    -------
    np.ndarray
        CLAHE-enhanced image (uint8).
    """
    _check_grayscale(image)
    img = _to_uint8(image)

    clahe = cv2.createCLAHE(clipLimit=float(clip_limit),
                            tileGridSize=tile_grid_size)
    return clahe.apply(img)



# 3) BBHE – Brightness Preserving Bi-Histogram Equalization
#    (Adapted from your BBHE notebook implementation)


def bbhe(image: np.ndarray, levels: int = 256) -> np.ndarray:
    """
    Brightness Preserving Bi-Histogram Equalization (BBHE).

    Algorithm (short):
    - Compute mean intensity.
    - Split histogram into [0, mean] and [mean+1, levels-1].
    - Equalize each part separately.
    - Combine with a piecewise mapping that preserves overall brightness.

    Parameters
    ----------
    image : np.ndarray
        2D grayscale image.
    levels : int
        Number of gray levels (usually 256 for uint8).

    Returns
    -------
    np.ndarray
        BBHE-enhanced image (uint8).
    """
    _check_grayscale(image)
    img = _to_uint8(image)
    h, w = img.shape

    mean_val = int(round(float(img.mean())))
    mean_val = max(0, min(levels - 1, mean_val))

    # Masks for lower and upper parts
    lower_mask = (img <= mean_val)
    upper_mask = (img > mean_val)

    # Histogram for whole image
    hist, _ = np.histogram(img.flatten(), bins=levels, range=(0, levels))

    # Lower part histogram
    hist_l = hist[: mean_val + 1]
    # Upper part histogram
    hist_u = hist[mean_val + 1 :]

    # Pixel counts
    count_l = float(hist_l.sum())
    count_u = float(hist_u.sum())

    # Avoid division by zero
    if count_l == 0 or count_u == 0:
        # Fall back to global HE
        return histogram_equalization(img)

    # PDFs
    pdf_l = hist_l / count_l
    pdf_u = hist_u / count_u

    # CDFs
    cdf_l = np.cumsum(pdf_l)
    cdf_u = np.cumsum(pdf_u)

    # Build LUT
    lut = np.zeros(levels, dtype=np.float32)

    # Lower mapping [0, mean_val]
    for i in range(0, mean_val + 1):
        lut[i] = mean_val * cdf_l[i]

    # Upper mapping [mean_val+1, levels-1]
    low_u = mean_val + 1
    high_u = levels - 1
    span_u = max(1, (high_u - low_u))
    for i in range(low_u, levels):
        lut[i] = low_u + span_u * cdf_u[i - low_u]

    # Apply LUT
    out = lut[img]
    return out.clip(0, 255).astype(np.uint8)


# 4) DHE – Dynamic Histogram Equalization
#    Here implemented as a multi-peak HE (MPHE-style) based on your MPHE code.

def dhe_multi_peak(
    image: np.ndarray,
    num_bins: int = 256,
    peak_height: float = 200000.0,
) -> np.ndarray:
    """
    Dynamic / Multi-Peak Histogram Equalization (MPHE-style DHE).

    Idea:
    - Compute the histogram of the image.
    - Detect significant peaks (local maxima above `peak_height`).
    - Use those peaks to split the intensity range into sub-ranges.
    - Apply standard HE inside each sub-range separately.

    Parameters
    ----------
    image : np.ndarray
        2D grayscale image.
    num_bins : int
        Number of bins for histogram (usually 256).
    peak_height : float
        Minimum height to consider a peak.

    Returns
    -------
    np.ndarray
        DHE-enhanced image (uint8).
    """
    _check_grayscale(image)
    img = _to_uint8(image)

    # Histogram
    hist, _ = np.histogram(img.flatten(), num_bins, [0, num_bins])

    # Find peaks
    peaks = _find_peaks_1d(hist, min_height=peak_height)
    num_peaks = len(peaks)

    if num_peaks == 0:
        # If no strong peaks, just fall back to global HE
        return histogram_equalization(img)

    # Define intensity intervals using midpoints between peaks
    intervals = np.zeros((num_peaks + 1,), dtype=int)
    intervals[0] = 0
    intervals[-1] = num_bins - 1

    for i in range(num_peaks - 1):
        mid = (peaks[i] + peaks[i + 1]) // 2
        intervals[i + 1] = mid

    out = np.zeros_like(img)

    # Equalize each interval separately
    for i in range(num_peaks + 1):
        if i == 0:
            mask = (img <= intervals[i + 1])
        elif i == num_peaks:
            mask = (img > intervals[i])
        else:
            mask = (img > intervals[i]) & (img <= intervals[i + 1])

        if not np.any(mask):
            continue

        region = img[mask]
        eq_region = cv2.equalizeHist(region)
        out[mask] = eq_region

    return out


# 5) RMHE – Range-Modified / Weighted HE
#    Implemented using your Weighted Mean Separated HE (WMSHE) style.

def rmhe_weighted(
    image: np.ndarray,
    num_bins: int = 256,
    alpha: float = 1.2,
) -> np.ndarray:
    """
    Range-modified / Weighted Mean Separated Histogram Equalization (RMHE).

    Implementation is based on a weighted PMF and its cumulative distribution.

    Parameters
    ----------
    image : np.ndarray
        2D grayscale image.
    num_bins : int
        Histogram bins (for uint8, 256).
    alpha : float
        Exponent for the weighted PMF (alpha > 1 emphasizes higher probabilities).

    Returns
    -------
    np.ndarray
        RMHE-enhanced image (uint8).
    """
    _check_grayscale(image)
    img = _to_uint8(image)

    # Histogram and PMF
    hist, _ = np.histogram(img.flatten(), num_bins, [0, num_bins])
    total = float(hist.sum())
    if total == 0:
        return img.copy()

    pmf = hist / total

    # Weighted PMF and CDF
    wp = pmf ** alpha
    wcdf = np.cumsum(wp)

    # First equalization using weighted CDF
    img_eq = np.interp(img.flatten(), np.arange(num_bins), wcdf * (num_bins - 1))
    img_eq = img_eq.reshape(img.shape)

    # Histogram of equalized image
    hist_eq, _ = np.histogram(img_eq.flatten(), num_bins, [0, num_bins])
    cdf_eq = np.cumsum(hist_eq)

    # Normalize CDF of equalized image to [0, 255]
    cdf_eq = (cdf_eq - cdf_eq.min()).astype(np.float64)
    denom = cdf_eq.max()
    if denom > 0:
        cdf_eq = cdf_eq * 255.0 / denom

    # Final mapping using normalized CDF
    img_eq_norm = np.interp(img_eq.flatten(), np.arange(num_bins), cdf_eq)
    img_eq_norm = img_eq_norm.reshape(img.shape)

    return img_eq_norm.clip(0, 255).astype(np.uint8)


# 6) CLAHE + Static JND

def clahe_static_jnd(
    image: np.ndarray,
    jnd_map: np.ndarray,
    tile_grid_size: Tuple[int, int] = (8, 8),
    scale: float = 0.15,
    clip_range: Tuple[float, float] = (1.0, 5.0),
) -> np.ndarray:
    """
    CLAHE with a static clip limit derived from a global JND value.

    Steps:
    -------
    1. Take the mean JND over the entire image
    2. Convert it to a clip limit using: clip = scale * mean_jnd
    3. Clamp the clip limit to clip_range (e.g., 1.0–5.0)
    4. Apply CLAHE using this (static) clip limit

    Parameters
    ----------
    image : np.ndarray
        Input image (2D or 3D).
    jnd_map : np.ndarray
        Full JND map, same size as image.
    tile_grid_size : (int, int)
        CLAHE tile grid size.
    scale : float
        Scaling factor to convert JND to clip limit.
    clip_range : (float, float)
        Allowed range for clip limit (min, max).

    Returns
    -------
    np.ndarray
        CLAHE-enhanced image.
    """

    _check_grayscale(image)
    img = _to_uint8(image)

    if jnd_map.shape != img.shape:
        raise ValueError(
            f"JND shape {jnd_map.shape} must match image shape {img.shape}"
        )

    # 1) global JND
    mean_jnd = float(np.mean(jnd_map))

    # 2) map to clip limit
    clip = mean_jnd * scale

    # 3) clamp it
    clip_min, clip_max = clip_range
    clip = max(clip_min, min(clip_max, clip))

    # 4) apply CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=float(clip),
        tileGridSize=tile_grid_size,
    )

    return clahe.apply(img)


def clahe_dynamic_jnd(
    image: np.ndarray,
    jnd_map: np.ndarray,
    tile_size: Tuple[int, int] = (8, 8),
    clip_min: float = 1.0,
    clip_max: float = 5.0,
    invert_and_offset: bool = True,
    offset: float = 6.0,
) -> np.ndarray:
    """
    CLAHE with tile-wise clip limit controlled by a JND map.

    - JND map is scaled into [clip_min, clip_max].
    - For each tile, the mean JND in that tile is used to set the local clip limit.
    - Optionally, the clip limit is inverted and shifted like in your original
      dynamic JND CLAHE implementation: clip = -mean_jnd + offset.

    Parameters
    ----------
    image : np.ndarray
        2D grayscale image.
    jnd_map : np.ndarray
        2D JND map with the same shape as `image`.
    tile_size : (int, int)
        Tile height and width in pixels (used for slicing, and also passed to
        OpenCV as tileGridSize for CLAHE, following your original script).
    clip_min : float
        Minimum clip limit after scaling.
    clip_max : float
        Maximum clip limit after scaling.
    invert_and_offset : bool
        If True, use: clip = -mean_jnd + offset (clamped to [clip_min, clip_max]).
        If False, use the scaled mean_jnd directly as clip limit.
    offset : float
        Offset used when `invert_and_offset` is True.

    Returns
    -------
    np.ndarray
        CLAHE-enhanced image (uint8).
    """
    _check_grayscale(image)
    img = _to_uint8(image)

    jnd = np.asarray(jnd_map, dtype=np.float32)
    if jnd.shape != img.shape:
        raise ValueError(
            f"jnd_map shape {jnd.shape} must match image shape {img.shape}."
        )

    # Scale JND map to [clip_min, clip_max]
    old_min = float(jnd.min())
    old_max = float(jnd.max())

    if old_max > old_min:
        scaled = ((jnd - old_min) / (old_max - old_min)) * (clip_max - clip_min) + clip_min
    else:
        # Degenerate case: constant JND map
        scaled = np.full_like(jnd, (clip_min + clip_max) / 2.0, dtype=np.float32)

    tile_h, tile_w = tile_size
    H, W = img.shape
    tile_rows = int(np.ceil(H / tile_h))
    tile_cols = int(np.ceil(W / tile_w))

    output = np.zeros_like(img)
    clahe = cv2.createCLAHE(tileGridSize=tile_size)

    for i in range(tile_rows):
        for j in range(tile_cols):
            r0, r1 = i * tile_h, min((i + 1) * tile_h, H)
            c0, c1 = j * tile_w, min((j + 1) * tile_w, W)

            tile_img = img[r0:r1, c0:c1]
            if tile_img.size == 0:
                continue

            tile_jnd = scaled[r0:r1, c0:c1]
            mean_jnd = float(tile_jnd.mean())

            if invert_and_offset:
                clip = -mean_jnd + offset
            else:
                clip = mean_jnd

            # Clamp to [clip_min, clip_max]
            clip = max(clip_min, min(clip_max, clip))

            clahe.setClipLimit(float(clip))
            tile_eq = clahe.apply(tile_img)
            output[r0:r1, c0:c1] = tile_eq

    return output



# Optional: small demo when running this file directly


if __name__ == "__main__":
    # Example usage (you can remove this block if you want a purely library file).
    import os

    test_path = "example_input.png"
    if os.path.exists(test_path):
        img_in = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)

        he = histogram_equalization(img_in)
        cla = clahe_opencv(img_in)
        bb = bbhe(img_in)
        dhe = dhe_multi_peak(img_in)
        rm = rmhe_weighted(img_in)

        cv2.imwrite("example_he.png", he)
        cv2.imwrite("example_clahe.png", cla)
        cv2.imwrite("example_bbhe.png", bb)
        cv2.imwrite("example_dhe.png", dhe)
        cv2.imwrite("example_rmhe.png", rm)

        print("Demo images saved next to the script.")
    else:
        print(
            "Run this file with an 'example_input.png' in the same folder "
            "to generate demo outputs."
        )
