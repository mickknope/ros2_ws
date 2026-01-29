#!/usr/bin/env python3
import cv2
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
import mediapipe as mp
from collections import deque

class FingerPublisher(Node):
    def __init__(self):
        super().__init__('finger_publisher')

        # -------- ROS2 --------
        self.pub = self.create_publisher(Int32, 'finger_count', 10)

        # -------- Camera (Low latency) --------
        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # สำคัญมาก

        # -------- MediaPipe --------
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        self.mp_draw = mp.solutions.drawing_utils

        # -------- Stability filter (แทน smoothing เดิม) --------
        self.history = deque(maxlen=3)
        self.last_sent = -1

        # -------- Fast timer (~30 FPS) --------
        self.timer = self.create_timer(0.033, self.loop)

        self.get_logger().info("Finger control node (low latency) started")

    def count_fingers(self, lm):
        tips = [4, 8, 12, 16, 20]
        fingers = 0

        # Thumb (basic version)
        if lm.landmark[4].x < lm.landmark[3].x:
            fingers += 1

        # Other fingers
        for i in range(1, 5):
            if lm.landmark[tips[i]].y < lm.landmark[tips[i]-2].y:
                fingers += 1

        return fingers

    def loop(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = self.hands.process(rgb)

        finger_count = 0

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            finger_count = self.count_fingers(hand)

            self.mp_draw.draw_landmarks(
                frame, hand, self.mp_hands.HAND_CONNECTIONS
            )

        # -------- stability check (ไม่หน่วง แต่กันสั่น) --------
        self.history.append(finger_count)

        if len(self.history) == self.history.maxlen:
            if all(v == self.history[0] for v in self.history):
                stable_value = self.history[0]

                if stable_value != self.last_sent:
                    msg = Int32()
                    msg.data = stable_value
                    self.pub.publish(msg)
                    self.last_sent = stable_value

        # -------- display --------
        cv2.putText(
            frame,
            f'Fingers: {finger_count}',
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3
        )

        cv2.imshow("Hand Control (Low Latency)", frame)
        cv2.waitKey(1)

def main():
    rclpy.init()
    node = FingerPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cap.release()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
