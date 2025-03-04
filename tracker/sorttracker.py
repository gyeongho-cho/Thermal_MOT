import numpy as np
from filterpy.kalman import KalmanFilter

class KalmanBoxTracker:
    def __init__(self, bbox):
        """
        Initialize a Kalman filter for tracking an object.
        bbox: [x1, y1, x2, y2, conf]
        """
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0],
                               [0, 1, 0, 0, 0, 1, 0],
                               [0, 0, 1, 0, 0, 0, 1],
                               [0, 0, 0, 1, 0, 0, 0],
                               [0, 0, 0, 0, 1, 0, 0],
                               [0, 0, 0, 0, 0, 1, 0],
                               [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0],
                               [0, 1, 0, 0, 0, 0, 0],
                               [0, 0, 1, 0, 0, 0, 0],
                               [0, 0, 0, 1, 0, 0, 0]])
        self.kf.x[:4] = np.array(bbox[:4]).reshape((4, 1))
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
    
    def predict(self):
        self.kf.predict()
        self.time_since_update += 1
        return self.kf.x[:4].flatten()
    
    def update(self, bbox):
        self.kf.update(np.array(bbox[:4]).reshape((4, 1)))
        self.time_since_update = 0
    
    def get_state(self):
        return self.kf.x[:4].flatten()

class SORTTracker:
    def __init__(self):
        self.trackers = []
        self.next_id = 0
    
    def update(self, yolo_detections):
        """
        Update trackers with new YOLO detections.
        """
        updated_tracks = []
        for tracker in self.trackers:
            tracker.predict()
        
        new_trackers = []
        for det in yolo_detections:
            matched = False
            for tracker in self.trackers:
                if np.linalg.norm(tracker.get_state()[:2] - np.array(det[:2])) < 50:
                    tracker.update(det)
                    updated_tracks.append(tracker.get_state().tolist() + [tracker.id])
                    matched = True
                    break
            if not matched:
                new_tracker = KalmanBoxTracker(det)
                new_trackers.append(new_tracker)
                updated_tracks.append(new_tracker.get_state().tolist() + [new_tracker.id])
        
        self.trackers = new_trackers + self.trackers
        return updated_tracks
