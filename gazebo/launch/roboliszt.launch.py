"""ROS 2 (Jazzy) + Gazebo Harmonic launch for RoboLiszt  -- NOT EXECUTED (no ROS/Gazebo in the authoring environment).

    ros2 launch roboliszt_gazebo roboliszt.launch.py                          # position mode (tasks 1-4)
    ros2 launch roboliszt_gazebo roboliszt.launch.py mode:=effort             # effort mode (task 5): spawn the controllers
    ros2 launch roboliszt_gazebo roboliszt.launch.py mode:=effort touch:=true finger:=middle profile:=pp

Requires ros_gz_sim, ros_gz_bridge, gz_ros2_control, controller_manager, joint_trajectory_controller,
forward_command_controller. The URDFs contain the token @CONTROLLERS_DIR@ (python gazebo/generate.py), replaced here
with the installed share directory (or the source tree when the package is not installed).
For task 5 generate the world with  python gazebo/generate.py --rtf 0.25  (the python node is not real-time at 1 kHz).
"""
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

SRC = Path(__file__).resolve().parents[1]


def _share() -> Path:
    try:
        from ament_index_python.packages import get_package_share_directory
        return Path(get_package_share_directory("roboliszt_gazebo"))
    except Exception:                                                    # not installed: run from the source tree
        return SRC


def _setup(context):
    share = _share()
    mode = LaunchConfiguration("mode").perform(context)
    urdf_name = "roboliszt_effort.urdf" if mode == "effort" else "roboliszt.urdf"
    urdf = (share / "urdf" / urdf_name).read_text().replace("@CONTROLLERS_DIR@", str(share / "config"))
    world = share / "worlds" / "piano.sdf"
    gz = ExecuteProcess(cmd=["gz", "sim", "-r", str(world)], output="screen")
    rsp = Node(package="robot_state_publisher", executable="robot_state_publisher", output="screen",
               parameters=[{"robot_description": urdf, "use_sim_time": True}])
    spawn = Node(package="ros_gz_sim", executable="create", output="screen",
                 arguments=["-topic", "robot_description", "-name", "roboliszt", "-allow_renaming", "true"])
    bridge = Node(package="ros_gz_bridge", executable="parameter_bridge", output="screen",
                  arguments=["/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
                             "/world/piano_world/model/piano/joint_state@sensor_msgs/msg/JointState[gz.msgs.Model"])
    ctrls = ["joint_state_broadcaster", "carriage_controller", "effort_controller"] if mode == "effort" else \
            ["joint_state_broadcaster", "hand_controller"]
    spawners = [Node(package="controller_manager", executable="spawner", output="screen",
                     arguments=[c, "--controller-manager", "/controller_manager"]) for c in ctrls]
    actions = [gz, rsp, bridge, TimerAction(period=3.0, actions=[spawn]), TimerAction(period=8.0, actions=spawners)]
    if mode == "effort" and LaunchConfiguration("touch").perform(context) == "true":
        actions.append(TimerAction(period=12.0, actions=[Node(
            package="roboliszt_gazebo", executable="task5_touch_node", output="screen",
            parameters=[{"use_sim_time": True, "finger": LaunchConfiguration("finger").perform(context),
                         "profile": LaunchConfiguration("profile").perform(context)}])]))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("mode", default_value="position", description="position | effort"),
        DeclareLaunchArgument("touch", default_value="false", description="start task5_touch_node (effort mode)"),
        DeclareLaunchArgument("finger", default_value="middle"),
        DeclareLaunchArgument("profile", default_value="pp"),
        OpaqueFunction(function=_setup),
    ])
