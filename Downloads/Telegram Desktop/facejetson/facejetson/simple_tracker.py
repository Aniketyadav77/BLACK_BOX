import numpy as np

class Track:
    def __init__(self, bbox, track_id, name="Unknown"):
        self.bbox = bbox
        self.id = track_id
        self.name = name
        self.missed_frames = 0

class SimpleTracker:
    def __init__(self, iou_thresh=0.3, max_missed=10):
        self.tracks = []
        self.next_id = 0
        self.iou_thresh = iou_thresh
        self.max_missed = max_missed

    @staticmethod
    def iou(bb1, bb2):
        x1 = max(bb1[0], bb2[0])
        y1 = max(bb1[1], bb2[1])
        x2 = min(bb1[2], bb2[2])
        y2 = min(bb1[3], bb2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
        area2 = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])
        
        return intersection / (area1 + area2 - intersection + 1e-6)

    def update(self, detections):
        # detections list of dicts: {'bbox':..., 'name':...}
        updated_tracks = []
        
        for det in detections:
            best_id = None
            best_iou = 0
            
            for t in self.tracks:
                curr_iou = self.iou(det['bbox'], t.bbox)
                if curr_iou > best_iou and curr_iou > self.iou_thresh:
                    best_iou = curr_iou
                    best_id = t.id
            
            if best_id is not None:
                # Update existing track
                track = next(t for t in self.tracks if t.id == best_id)
                track.bbox = det['bbox']
                track.missed_frames = 0
                if det.get('name') != "Unknown":
                    track.name = det['name']
                updated_tracks.append(track)
            else:
                # New track
                new_track = Track(det['bbox'], self.next_id, det.get('name', 'Unknown'))
                self.next_id += 1
                updated_tracks.append(new_track)

        # Handle lost tracks (keep them for a few frames)
        matched_ids = [t.id for t in updated_tracks]
        for t in self.tracks:
            if t.id not in matched_ids:
                t.missed_frames += 1
                if t.missed_frames < self.max_missed:
                    updated_tracks.append(t)
                    
        self.tracks = updated_tracks
        return self.tracks
