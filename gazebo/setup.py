from glob import glob

from setuptools import setup

package_name = "roboliszt_gazebo"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/urdf", glob("urdf/*.urdf")),
        ("share/" + package_name + "/worlds", glob("worlds/*.sdf")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
        ("share/" + package_name + "/meshes", glob("meshes/*.obj")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    entry_points={"console_scripts": [
        "task_runner = roboliszt_gazebo.task_runner:main",
        "task5_touch_node = roboliszt_gazebo.task5_touch_node:main",
        "piano_sound_node = roboliszt_gazebo.piano_sound_node:main",
    ]},
)
