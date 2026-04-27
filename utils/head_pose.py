import numpy as np
import cv2



def get_image_points(kpts):

    kpts = kpts.cpu().numpy()
    left_eye = kpts[0]
    right_eye = kpts[1]
    nose  = kpts[2]
    left_mouth =kpts[3]
    right_mouth = kpts[4]

    image_points = np.array([nose,left_eye,right_eye,left_mouth,right_mouth],dtype='double')
    return image_points


def get_model_points():
    model_points = np.array([   (0.0, 0.0, 0.0),          # Nose
 
    (-30.0, 30.0, -30.0),     # Left Eye
    (30.0, 30.0, -30.0),      # Right Eye
    (-25.0, -30.0, -30.0),    # Left Mouth
    (25.0, -30.0, - 30.0) ],dtype='double')
    return model_points

def camera_mat(frame):
    h, w = frame.shape[:2]

    focal_length = w
    center = (w/2, h/2)

    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")
    return camera_matrix

def estimate_head_pose(kpts,frame):
    model_points = get_model_points()
    image_points = get_image_points(kpts)
    camera_matrix = camera_mat(frame)

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        None,
        flags=cv2.SOLVEPNP_EPNP
    )

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)
    print(rotation_matrix) 
    pitch = angles[0]
    yaw = angles[1]
    roll = angles[2]
    if pitch > 90:
        pitch -= 180
    elif pitch < -90:
        pitch += 180


    return pitch,yaw,roll
