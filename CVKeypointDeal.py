import cv2



def keypoints_to_tuple(keypoints):
    return [(kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id) for kp in keypoints]

def tuple_to_keypoints(keypoints_tuple):
    return [cv2.KeyPoint(x=x, y=y, size=size, angle=angle, response=response, octave=octave, class_id=class_id) for ((x, y), size, angle, response, octave, class_id) in keypoints_tuple]