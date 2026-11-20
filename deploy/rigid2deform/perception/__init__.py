"""rigid2deform's perception: the nodes that run in a camera environment and publish what the policy
reads. main.py starts the one config.yaml's `mode` needs (../mode.py).

| file                         | what                                                                               |
|------------------------------|------------------------------------------------------------------------------------|
| main.py                      | the entry: picks and starts one of the nodes below (or deploy/nodes/perception.py) |
| cotracker_node.py            | cotracker: one camera, CoTracker3 on the cloth -> object_points                    |
| cotracker_stereo_node.py     | cotracker, two cameras: tracked in both, triangulated -> object_points             |
| cotracker_stereo_picker.py   |   its window to find camera A's clicked points in camera B                         |
| cotracker_stereo_geometry.py |   its two-camera geometry (project, triangulate, align camera B)                   |
| pc_node.py                   | a point-cloud policy's: both D415s -> the live cloud -> point_cloud                |
| pc_ref_goal.py               |   its start: the reference or goal from the flat cloth and one click               |
| fake_sheet.py                | perception.source=fake, keypoint policy: a still sheet, no camera                  |
| pc_fake_node.py              | perception.source=fake, point-cloud policy: a still flat sheet's cloud             |

What the policy does with them is not here: ../sheets/ (object_poses / object_points -> the sheet's
particles, keypoint policy), ../pc_obs.py (point_cloud -> the network's input).
"""
