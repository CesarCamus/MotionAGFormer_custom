import numpy as np
import argparse


def filter_confident_keypoints(kpts_2d: np.ndarray, threshold: float):
    """
    Filters frames where all keypoints have a confidence score above the given threshold.

    Parameters:
    - kpts_2d (np.ndarray): Input keypoints with shape (1, num_frames, 17, 3)
    - threshold (float): Confidence threshold for filtering

    Returns:
    - valid_ids (List[int]): List of frame indices with all keypoints above threshold
    - filtered_kpts (np.ndarray): Filtered keypoints with shape (N_valid, 17, 3)
    """
    # Remove first singleton dimension: (num_frames, 17, 3)

    # Extract confidence values: shape (num_frames, 17)
    confidences = kpts_2d[..., 2]

    # Identify frames where all keypoints have confidence > threshold
    valid_mask = np.all(confidences > threshold, axis=1)

    # Get indices of valid frames
    valid_ids = np.where(valid_mask)[0].tolist()

    # Filter keypoints for valid frames
    filtered_kpts = kpts_2d[valid_mask]

    return valid_ids, filtered_kpts

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Filter keypoints based on confidence threshold.")
    parser.add_argument("--npz_file", type=str, help="Path to the .npz file containing keypoints.")
    parser.add_argument("--threshold", type=float, help="Confidence threshold for filtering.")
    args = parser.parse_args()

    # Load the .npz file
    data = np.load(args.npz_file)
    kpts_2d = data["reconstruction"].squeeze()  # Remove singleton dimension if present
    print("Loaded keypoints shape:", kpts_2d.shape)

    # Filter keypoints
    valid_ids, filtered_kpts = filter_confident_keypoints(kpts_2d, args.threshold)

    # Print results
    print("Valid frame indices:", valid_ids)
    print("Filtered keypoints shape:", filtered_kpts.shape)