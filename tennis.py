from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
from court import Court
from players_statistics import Statistics
from ball_tracker import BallTracker
import torch
import argparse
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser("Tennis analyst")
    parser.add_argument("--player_detector", "-p",type=str, required=True, default="best_yolo.pt",
                        help= "Player detect model path")
    parser.add_argument("--threshold", "-t", type=float, default=0.5,
                        help="Player model confidence score threshold")
    parser.add_argument("--ball_detector", "-b", type=str, required=True, default="best_tracknet.pt",
                        help="Ball detect model path")

    parser.add_argument("--input_vid", "-i", type=str, required=True, default="input_video.mp4",
                        help="Input video path")
    parser.add_argument("--output_vid", "-o", type=str, required=True, default="out.mp4",
                        help="Output video path")

    args = parser.parse_args()
    return args

def ball_interpolate(trackers):
    df = pd.DataFrame(trackers)
    df.interpolate(method='polynomial',order=2, inplace= True)
    df = df.bfill().ffill()
    return df.to_numpy()

def draw_player(properties, original_frame):
    xmin,ymin,xmax,ymax, player_id = [int(prop) for prop in properties]
    cv2.rectangle(
        original_frame,
        (xmin, ymin),
        (xmax, ymax),
        color=(0,0,255),
        thickness=1,
        lineType=1
    )
    cv2.putText(
        original_frame,
        "Player ID: {}".format(player_id),
        (xmin, ymax),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.4,
        color= (255, 255, 255),
        thickness=1,
    )
    return original_frame

def draw_ball(tracker, idx, original_frame, trace=4):
    for i in range(trace):
        if idx > i:
            if tracker[idx - i][0]:
                x = int(tracker[idx - i][0])
                y = int(tracker[idx - i][1])
                cv2.circle(original_frame, (x, y), radius=1, color=(0, 255, 0), thickness=12 - i)
            else:
                break
    return original_frame


def speed_estimator(players_tracker, fps, frame_index, id):
    court = Court()
    xmin,ymin,xmax,ymax,_ = players_tracker[frame_index-1][id]
    x_prev_px = (xmin+xmax)/2
    y_prev_px = ymin
    x_prev, y_prev = court.perspective_transform(x_prev_px, y_prev_px)[0][0]

    xmin,ymin,xmax,ymax,_ = players_tracker[frame_index][id]
    x_new_px = (xmin+xmax)/2
    y_new_px = ymin
    x_new, y_new = court.perspective_transform(x_new_px, y_new_px)[0][0]

    dis_real = np.sqrt((x_new-x_prev)**2 + (y_new-y_prev)**2)
    time = 1/fps

    speed = dis_real/time
    return speed*3.6

def player_speed(players_tracker, fps, frame_index, id):
    speed = []
    for i in range(frame_index-5, frame_index):
        esti_speed = speed_estimator(players_tracker, fps, i, id)
        if esti_speed < 30:
            speed.append(esti_speed)

    return np.mean(speed) if speed else 0


if __name__ == '__main__':
    args = get_args()

    detect_model = YOLO(args.player_detector)
    detect_model.conf = args.threshold

    ball_tracker = BallTracker(args.ball_detector)

    cap = cv2.VideoCapture(args.input_vid)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        args.output_vid,
        cv2.VideoWriter_fourcc(*"MP4V"),
        fps,
        (width, height)
    )

    x1, y1, x2, y2 = 200, 100, 1720, 1050

    frames = []
    frames_num = 0
    players_trackers = []
    ball_trackers = []

    print("Detect Players.............")

    while cap.isOpened():
        flag, frame = cap.read()
        if not flag:
            break

        # Detect players
        roi = frame[y1:y2, x1:x2]
        prediction = detect_model.track(roi, persist=True)[0]

        categories = prediction.boxes.cls.cpu().numpy()
        boxes = prediction.boxes.xyxy.cpu().numpy()
        boxes[:, [0, 2]] += x1  # shift xmin, xmax
        boxes[:, [1, 3]] += y1  # shift ymin, ymax
        conf = prediction.boxes.conf.cpu().numpy()
        object_id = prediction.boxes.id.cpu().numpy()

        players = []

        for idx in range(len(categories)):
            category = int(categories[idx])
            xmin, ymin, xmax, ymax = boxes[idx]

            if category ==  1:
                player_id = object_id[idx]
                players.append([xmin, ymin, xmax, ymax, player_id])

        if len(players) > 1 and players[0][1] > players[1][1]:
            players[0], players[1] = players[1], players[0]
        players_trackers.append(players)

        frames.append(frame)
        frames_num +=1

    print("Detect Ball.............")
    progressbar = tqdm(range(frames_num), colour="cyan")

    for frame_idx in progressbar:
        # Ball tracks
        if frame_idx == 0:
            xcenter, ycenter = ball_tracker.get_prediction(frames[0], frames[0], frames[0])

        elif frame_idx == 1:
            xcenter, ycenter = ball_tracker.get_prediction(frames[0], frames[0], frames[1])

        else:
            xcenter, ycenter = ball_tracker.get_prediction(frames[frame_idx-2], frames[frame_idx-1], frames[frame_idx])

        ball_trackers.append([xcenter, ycenter])
        progressbar.set_postfix({"Detected": f"{frame_idx+1}/{frames_num+1}"})

    # Ball interpolate
    ball_trackers = ball_interpolate(ball_trackers)

    # Player statistics
    stat = Statistics()

    print("Draw annotations.............")
    progress_bar = tqdm(range(frames_num), colour="yellow")

    #Draw annotations
    for index in progress_bar:
        # Draw players annotations
        players = players_trackers[index]
        for player_properties in players:
            frames[index] = draw_player(player_properties, frames[index])

        # Draw ball annotations
        frames[index] = draw_ball(ball_trackers, index, frames[index])

        if index > 0 and index % 5 == 0:
            player1_speed = player_speed(players_trackers, fps, index, 0)
            player2_speed = player_speed(players_trackers, fps, index, 1)

            stat.stats["player1_speed"] = player1_speed
            stat.stats["player2_speed"] = player2_speed

            stat.stats["player1_average_speed"].append(player1_speed)
            stat.stats["player2_average_speed"].append(player2_speed)

        frames[index] = stat.draw_stats_box(frames[index])
        frames[index] = stat.draw_stats(frames[index])

        progress_bar.set_postfix({"Draw": f"{frame_idx+1}/{frames_num+1}"})


    for index in range(frames_num):
        out.write(frames[index])

    cap.release()
    out.release()
