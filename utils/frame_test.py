import cv2
import numpy as np

def get_embedding():
    return np.random.rand(128).astype('float32')

def blur_score(img):
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray,cv2.CV_64F).var()


import cv2

def align_face(img, roll, center=None):
    h, w = img.shape[:2]

    if center is None:
        center = (w // 2, h // 2)

    # Rotate using roll angle
    M = cv2.getRotationMatrix2D(center, roll, 1.0)

    aligned = cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=cv2.INTER_CUBIC
    )

    return aligned

def is_duplicate(new_emb, stored_embeddings, threshold=0.6):
    """
    Returns True if embedding is too similar to existing ones.
    """

    for emb in stored_embeddings:
        similarity = np.dot(new_emb, emb)
        if similarity > threshold:
            return True

    return False
def filter_frame(candidates,yaw,pitch,face_crop,roll):### many  params optimize it!!!!
    if abs(yaw) <= 70 and abs(pitch) <= 30:

        blur = blur_score(face_crop)

        if blur > 120:

            aligned = align_face(face_crop,roll)
            aligned = cv2.resize(aligned, (112,112))

            emb = get_embedding()
            emb = emb / np.linalg.norm(emb)

            candidates.append((emb, blur, aligned))
    return candidates
def select_candidates(candidates):
    candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
    selected = []
    selected_embs = []
    for emb,blur,img in candidates:
        if len(selected)>10:
            break
        if not is_duplicate(emb,selected_embs,threshold = 0.9):
            selected.append(img)
            selected_embs.append(emb)
    return selected,selected_embs