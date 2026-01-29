import cv2
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import mediapipe as mp
from collections import deque

class FingerCamCmdVel(Node):
    def __init__(self):
        super().__init__('finger')

        # ---------- ROS2 ----------
        self.pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # ---------- Camera ----------
        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # ลดดีเลย์

        # ---------- MediaPipe ----------
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils

        # ---------- Smoothing ----------
        self.history = deque(maxlen=3)
        self.last_sent = -1

        # ---------- Timer ----------
        self.timer = self.create_timer(0.1, self.loop)  # ~10 Hz

        self.get_logger().info("Finger + Camera + cmd_vel started")

    def count_fingers(self, hand):
        tips = [4, 8, 12, 16, 20]
        fingers = 0

        # Thumb (สำหรับมือขวา + flip แล้ว)
        if hand.landmark[tips[0]].x < hand.landmark[3].x:
            fingers += 1

        # Other fingers
        for i in range(1, 5):
            if hand.landmark[tips[i]].y < hand.landmark[tips[i]-2].y:
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

        # ---------- smoothing ----------
        self.history.append(finger_count)
        finger_count = round(sum(self.history) / len(self.history))

        # ---------- publish cmd_vel ----------
        if finger_count != self.last_sent:
            twist = Twist()

            if finger_count == 0:          # กำมือ
                twist.linear.x = 0.0
                twist.angular.z = 0.0

            elif finger_count == 1:        # เดินหน้า
                twist.linear.x = 0.2

            elif finger_count == 2:        # ถอย
                twist.linear.x = -0.2

            elif finger_count == 3:        # เลี้ยวซ้าย
                twist.angular.z = 0.8

            elif finger_count == 4:        # เลี้ยวขวา
                twist.angular.z = -0.8

            else:
                twist.linear.x = 0.0
                twist.angular.z = 0.0

            self.pub.publish(twist)
            self.last_sent = finger_count

        # ---------- debug ----------
        cv2.putText(frame, f'Fingers: {finger_count}',
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 255, 0), 3)

        cv2.imshow("Finger Control", frame)
        cv2.waitKey(1)

def main():
    rclpy.init()
    node = FingerCamCmdVel()
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
