import cv2
import numpy as np
import onnxruntime as ort

session = ort.InferenceSession(r"E:\keyur\python\project\cctv face recognition\arc.onnx")

def get_embedding(face_img):
    face = cv2.resize(face_img, (112, 112))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    # face = np.transpose(face, (2, 0, 1))
    face = np.expand_dims(face, axis=0).astype(np.float32)

    input_name = session.get_inputs()[0].name
    embedding = session.run(None, {input_name: face})[0][0]

    # normalize
    embedding = embedding / np.linalg.norm(embedding)

    return embedding

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
    if abs(yaw) <= 40 and abs(pitch) <= 25:

        blur = blur_score(face_crop)

        if blur > 100:

            aligned = align_face(face_crop,roll)
            aligned = cv2.resize(aligned, (112,112))


            candidates.append((blur, aligned))
    return candidates
def select_candidates(candidates):
    candidates = sorted(candidates, key=lambda x: float(x[0]), reverse=True)
    selected = []
    selected_embs = []
    for blur,img in candidates:
        if len(selected)>=10:
            break
        emb = get_embedding(face_img=img)
        # emb = emb / np.linalg.norm(emb)
        if  not is_duplicate(emb,selected_embs,threshold = 0.95):
            selected.append(img)
            selected_embs.append(emb)
    return selected,selected_embs