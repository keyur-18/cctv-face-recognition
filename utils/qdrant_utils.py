from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import numpy as np
import uuid
from qdrant_client.models import Filter, FieldCondition, MatchValue


class FaceDB:
    def __init__(self, dim=512, collection_name="faces"):
        self.dim = dim
        self.collection_name = collection_name

        # connect (local DB file)
        self.client = QdrantClient(path="./qdrant_data")

        # create collection if not exists
        if collection_name not in [c.name for c in self.client.get_collections().collections]:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=dim,
                    distance=Distance.COSINE 
                )
            )

    def add_embeddings(self, name, embeddings):
        points = []

        for emb in embeddings:
            emb = np.array(emb).astype('float32')

            # normalize (important for cosine)
            emb = emb / np.linalg.norm(emb)

            point_id = str(uuid.uuid4())

            points.append(
                PointStruct(
                    id=point_id,
                    vector=emb.tolist(),
                    payload={"name": name}
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def search(self, embedding):
        embedding = np.array(embedding).astype('float32')
        embedding = embedding / np.linalg.norm(embedding)
    
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=embedding.tolist(),
            limit=1
        )
    
        points = results.points
    
        if len(points) == 0:
            return None, 0.0
    
        return points[0].payload["name"], points[0].score
    
    def name_exists(self,name):
        results,_  = self.client.scroll(collection_name=self.collection_name,
                                        scroll_filter=Filter(
                                            must = [FieldCondition(
                                                key = "name",match=MatchValue(value=name))]),
                                        limit=1)
        return len(results)>0
    def delete_by_name(self, name):
     exist = self.name_exists(name)
     if  not exist : return f"{name} not exists!!!" 
     self.client.delete(
         collection_name=self.collection_name,
         points_selector=Filter(
             must=[
                 FieldCondition(
                     key="name",
                     match=MatchValue(value=name)
                 )
             ] 
         )
     )
     return f"{name} deleted successfully"

    def reset(self):
        """Delete full collection (like clearing FAISS index)"""
        self.client.delete_collection(self.collection_name)


    def close(self):
        self.client.close()