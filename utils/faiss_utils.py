import faiss
import numpy as np
import pickle
import os
class FaceDB:
    def __init__(self,dim = 128,index_path = "faces.index",label_path = "label_pkl"):
        self.dim=dim
        self.index_path = index_path
        self.label_path = label_path

        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            with open(label_path, "rb") as f:
                self.labels = pickle.load(f)
        else:
            self.index = faiss.IndexFlatIP(dim)  # cosine similarity
            self.labels = []

    def add_embeddings(self, name, embeddings):
            embeddings = np.array(embeddings).astype('float32')

            # normalize
            faiss.normalize_L2(embeddings)

            self.index.add(embeddings)

            # store labels
            self.labels.extend([name] * len(embeddings))

    def search(self, embedding, threshold=0.6):
            embedding = np.array(embedding).astype('float32').reshape(1, -1)

            faiss.normalize_L2(embedding)

            D, I = self.index.search(embedding, k=1)

            score = D[0][0]
            idx = I[0][0]

            if idx == -1 or score < threshold:
                return "Unknown", score

            return self.labels[idx], score
    def save(self):
            faiss.write_index(self.index, self.index_path)
            with open(self.label_path, "wb") as f:
                pickle.dump(self.labels, f)