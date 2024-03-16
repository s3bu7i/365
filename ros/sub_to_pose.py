from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

import rclpy  # ROS 2 Python API
from nav2_simple_commander.robot_navigator import BasicNavigator  # A api for send goal to nav2

# Assignment evaluation criteria:
# - CPU and memory usage
# - Code quality
# - Code organization
# - Code comments
# - Code performance
# - Code style guide


class PoseSubscriber(Node):
    """
    A ROS 2 node for subscribing to PoseStamped messages and passing coordinates to a BasicNavigator.
    """

    def __init__(self):
        """
        Initialize the PoseSubscriber node.
        """
        super().__init__("pose_subscriber")
        self.subscription = self.create_subscription(PoseStamped, "pose", self.listener_callback, 10)
        self.basic_navigator = BasicNavigator()
        self.pose_list = []
        self.current_pose = None

    def listener_callback(self, msg):
        """
        Callback function for processing incoming PoseStamped messages.

        Args:
            msg (PoseStamped): The received PoseStamped message.
        """

    def keyboard_input(self):
        """
        Get keyboard input from the user.

        Returns:
            str: The user's keyboard input.
        """
        positions = []
        while True:
            position = input("Enter position (x y z) or 'q' to quit: ")
            if position == 'q':
                break
            positions.append(position)
        return positions


if __name__ == "__main__":
    # Initialize the ROS 2 node
    rclpy.init()
    node = PoseSubscriber()
    # Get keyboard input for positions
    positions = node.keyboard_input()
    print("Positions:", positions)
    # Spin the node
    rclpy.spin(node)
    # Shutdown ROS 2
    rclpy.shutdown()


