import cv2
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree

def remove_overlapping(keypoints1, keypoints2, threshold=2):
    keypoints1_array = np.array([kp.pt for kp in keypoints1])
    keypoints2_array = np.array([kp.pt for kp in keypoints2])

    distances = cdist(keypoints1_array, keypoints2_array)
    overlapping = np.any(distances < threshold, axis=1)
    non_overlapping_keypoints1 = [kp for i, kp in enumerate(keypoints1) if not overlapping[i]]

    return non_overlapping_keypoints1



def remove_overlapping_Kd(keypoints1, keypoints2, threshold=2):
    keypoints1_array = np.array([kp.pt for kp in keypoints1])
    keypoints2_array = np.array([kp.pt for kp in keypoints2])

    tree = cKDTree(keypoints2_array)
    distances, _ = tree.query(keypoints1_array, k=1)
    non_overlapping_indices = np.where(distances >= threshold)[0]
    non_overlapping_keypoints1 = [keypoints1[i] for i in non_overlapping_indices]

    return non_overlapping_keypoints1

