from tracknet import Tracknet
import torch
import numpy as np
import cv2

def binary_heatmap(heatmap, threshold):
    return np.where(heatmap >= threshold, 255, 0).astype(np.uint8)

def np_to_tensor(frame):
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

        self.model.load_state_dict(torch.load(self.checkpoint_path, map_location=torch.device("cpu")))
        self.model.eval()


    def get_prediction(self, frame0, frame1, frame2):
        stack_fr, self.previous_frames = preprocessing(frame0,frame1,frame2, self.previous_frames)

        with torch.inference_mode():
            out = self.model(stack_fr)[0].cpu().numpy()
            heatmap = np.argmax(out, axis=0)
            bina_heatmap = binary_heatmap(heatmap, threshold=127)

            circles = cv2.HoughCircles(bina_heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=50, param2=2, minRadius=2, maxRadius=7)
            x, y = np.nan, np.nan
            if circles is not None:
                if len(circles) == 1:
                    x = np.float32(circles[0][0][0] * 3)
                    y = np.float32(circles[0][0][1] * 3)

            return x,y

