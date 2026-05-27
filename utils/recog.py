from collections import defaultdict
import numpy as np
from frame_test import filter_frame, select_candidates

class RecognitionSystem:
    def __init__(self, db):
        self.db = db

        self.track_data = defaultdict(lambda: {
            "candidates": [],
            "votes": defaultdict(int),
            "done": False
        })

        self.marked_present = set()

    def process(self, track_id, yaw, pitch, face_crop, roll):
        data = self.track_data[track_id]

        if data["done"]:
            return None

     
        data["candidates"] = filter_frame(
            data["candidates"], yaw, pitch, face_crop, roll
        )



       
        _, selected_embs = select_candidates(data["candidates"])

   
        votes = data["votes"]  

  
        if(len(selected_embs)>0):
            label, score = self.db.search(selected_embs[0])
            if score > 0.93:   # threshold HERE
  
                return label
            else : None
        else : None
