import numpy as np
import cv2


class Statistics:
    def __init__(self):
        self.stats = {
            "player1_average_speed": [],
            "player2_average_speed": [],
            "player1_speed":0,
            "player2_speed":0
        }
        self.box_width = 350
        self.box_height = 230
        self.alpha = 0.7  # transparency


    def draw_stats_box(self, ori_frame):
        width, height = 350, 230

        frame_height, frame_width = ori_frame.shape[:2]

        # Position the box near the bottom-right corner
        start_x = frame_width - self.box_width - 50
        start_y = frame_height - self.box_height - 50
        end_x = start_x + self.box_width
        end_y = start_y + self.box_height

        # Copy the frame for overlay
        overlay = ori_frame.copy()

        # Draw filled rectangle on overlay
        cv2.rectangle(
            overlay,
            (start_x, start_y),
            (end_x, end_y),
            (0, 0, 0),  # Black
            -1
        )

        # Blend with transparency
        cv2.addWeighted(overlay, self.alpha, ori_frame, 1 - self.alpha, 0, ori_frame)

        return ori_frame


    def draw_stats(self,ori_frame):
        player_1_speed = self.stats["player1_speed"]
        player_2_speed = self.stats["player2_speed"]

        avg_player1_speed = np.mean(self.stats['player1_average_speed']) if self.stats["player1_average_speed"] else 0
        avg_player2_speed = np.mean(self.stats['player2_average_speed']) if self.stats["player2_average_speed"] else 0

        frame_height, frame_width = ori_frame.shape[:2]


        start_x = frame_width - self.box_width- 50
        start_y = frame_height - self.box_height - 50


        text = "     Player 1     Player 2"
        cv2.putText(ori_frame, text, (start_x + 80, start_y + 30),
                                                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        text = "Player Speed"
        ori_frame = cv2.putText(ori_frame, text, (start_x + 10, start_y + 120),
                                                 cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        text = f"{player_1_speed:.1f} km/h    {player_2_speed:.1f} km/h"
        ori_frame = cv2.putText(ori_frame, text, (start_x + 130, start_y + 120),
                                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        text = "avg. P. Speed"
        ori_frame = cv2.putText(ori_frame, text, (start_x + 10, start_y + 200),
                                                 cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        text = f"{avg_player1_speed:.1f} km/h    {avg_player2_speed:.1f} km/h"
        ori_frame = cv2.putText(ori_frame, text, (start_x + 130, start_y + 200),
                                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


        return ori_frame