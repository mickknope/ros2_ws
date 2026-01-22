import sys
import threading

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton,
    QVBoxLayout, QHBoxLayout
)

class RobotGUINode(Node):
    def __init__(self):
        super().__init__('robot_gui_node')
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

    def send_cmd(self, linear, angular):
        msg = Twist()
        msg.linear.x = linear
        msg.angular.z = angular
        self.cmd_pub.publish(msg)

class RobotGUI(QWidget):
    def __init__(self, ros_node):
        super().__init__()
        self.node = ros_node
        self.setWindowTitle("Micro-ROS Robot Controller")

        btn_fwd = QPushButton("Forward")
        btn_back = QPushButton("Backward")
        btn_left = QPushButton("Left")
        btn_right = QPushButton("Right")
        btn_stop = QPushButton("STOP")

        btn_fwd.clicked.connect(lambda: self.node.send_cmd(0.5, 0.0))
        btn_back.clicked.connect(lambda: self.node.send_cmd(-0.5, 0.0))
        btn_left.clicked.connect(lambda: self.node.send_cmd(0.0, 0.8))
        btn_right.clicked.connect(lambda: self.node.send_cmd(0.0, -0.8))
        btn_stop.clicked.connect(lambda: self.node.send_cmd(0.0, 0.0))

        layout = QVBoxLayout()
        layout.addWidget(btn_fwd)

        row = QHBoxLayout()
        row.addWidget(btn_left)
        row.addWidget(btn_stop)
        row.addWidget(btn_right)

        layout.addLayout(row)
        layout.addWidget(btn_back)

        self.setLayout(layout)

def ros_spin(node):
    rclpy.spin(node)

def main():
    rclpy.init()

    node = RobotGUINode()

    ros_thread = threading.Thread(target=ros_spin, args=(node,), daemon=True)
    ros_thread.start()

    app = QApplication(sys.argv)
    gui = RobotGUI(node)
    gui.show()

    app.exec_()

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()