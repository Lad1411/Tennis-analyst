from tracknet import Tracknet
import torch
import numpy as np
import cv2
from scipy.spatial.distance import euclidean

def binary_heatmap(heatmap, threshold):
    return np.where(heatmap >= threshold, 255, 0).astype(np.uint8)

def np_to_tensor(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = np.transpose(frame, axes=(2,0,1)) #(3, 1080, 1920)
    frame = np.stack([
        cv2.resize(frame[c], (640, 360), interpolation=cv2.INTER_AREA) for c in range(3)
    ])
    return torch.from_numpy(frame.astype(np.float32) / 255.0)

def preprocessing(frame0, frame1, frame2, prev_frames = None):
    # convert frames to tensor
    if prev_frames is not None:
        frame0_tensor, frame1_tensor = prev_frames
    else:
        frame0_tensor = np_to_tensor(frame0)
        frame1_tensor = np_to_tensor(frame1)

    frame2_tensor = np_to_tensor(frame2)

    next_cache = [frame1_tensor, frame2_tensor]

    stacked_frames = torch.cat((frame0_tensor, frame1_tensor, frame2_tensor), dim=0)
    stacked_frames = stacked_frames[None, ...]

    return stacked_frames, next_cache

class BallTracker:
    def __init__(self, checkpoint_path):
        self.model = Tracknet()
        self.checkpoint_path = checkpoint_path
        self.previous_frames = None

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.load_state_dict(torch.load(self.checkpoint_path, map_location=torch.device(self.device)))
        self.model.to(self.device)
        self.model.eval()

        self.prev_x = None
        self.prev_y = None


    def get_prediction(self, frame0, frame1, frame2):
        stack_fr, self.previous_frames = preprocessing(frame0,frame1,frame2, self.previous_frames)

        with torch.inference_mode():
            out = self.model(stack_fr.to(self.device))[0].cpu().numpy()
            heatmap = np.argmax(out, axis=0)
            bina_heatmap = binary_heatmap(heatmap, threshold=127)

            circles = cv2.HoughCircles(bina_heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=50, param2=5, minRadius=2, maxRadius=10)
            best_circle = np.nan
            x, y = np.nan, np.nan

            if circles is not None:
                circles = circles[0]
                if len(circles) == 1 or self.prev_x is None:
                    best_circle = circles[0]

                else:
                    mindist = float('inf')
                    for circle in circles:
                        x_pred = circle[0]*3
                        y_pred = circle[1]*3


                        dis = euclidean([self.prev_x, self.prev_y], [x_pred, y_pred])

                        if dis < mindist:
                            best_circle = circle
                            mindist = dis

            if best_circle is not np.nan:
                x = np.float32(best_circle[0]*3)
                y = np.float32(best_circle[1]*3)
                self.prev_x = x
                self.prev_y = y

            return x,y

