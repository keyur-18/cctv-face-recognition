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

        # 1. Collect good frames
        data["candidates"] = filter_frame(
            data["candidates"], yaw, pitch, face_crop, roll
        )

        # 2. Wait for enough samples
        # if len(data["candidates"]) < 15:
        #     return None

        # 3. Select best embeddings
        _, selected_embs = select_candidates(data["candidates"])

        # 4. Voting
        votes = data["votes"]  

        # for emb in selected_embs:
        #     label, score = self.db.search(emb)   
        if(len(selected_embs)>0):
            label, score = self.db.search(selected_embs[0])
            if score > 0.93:   # threshold HERE
            # votes[label] += 1
                return label
            else : None
        else : None
        # total_votes = sum(votes.values())

        # if total_votes == 0:
        #     data["done"] = True
        #     return "Unknown"

        # best_label = max(votes, key=votes.get)

        # if votes[best_label] / total_votes > 0.6:
        #     if best_label not in self.marked_present:
        #         print(f"✅ Present: {best_label}")
        #         self.marked_present.add(best_label)

        #     data["done"] = True
        #     return best_label

        # data["done"] = True
        # return "Unknown"