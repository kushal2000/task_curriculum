"""rigid2deform's perception: the nodes that run in a camera environment and publish what the policy
reads. main.py starts the one the config needs (policy.checkpoint's kind, object.name's perception).

| file               | what                                                                             |
|--------------------|----------------------------------------------------------------------------------|
| main.py            | the entry: picks and starts one of the nodes below (or deploy/nodes/perception.py) |
| cloth_tracker.py   | cotracker: one camera, CoTracker3 on the cloth -> object_points                  |
| stereo_tracker.py  | cotracker, two cameras: tracked in both, triangulated -> object_points           |
| stereo_picker.py   |   its window to find camera A's clicked points in camera B                       |
| stereo_geometry.py |   its two-camera geometry (project, triangulate, align camera B)                 |
| cloud_camera.py    | a point-cloud policy's: both D415s -> the live cloud -> point_cloud              |
| cloud_reference.py |   its start: the reference or goal from the flat cloth and one click             |
| fake_sheet.py      | perception.source=fake, keypoint policy: a still sheet, no camera                |
| fake_cloud.py      | perception.source=fake, point-cloud policy: a still flat sheet's cloud           |

What the policy does with them is not here: ../sheets/ (object_poses / object_points -> the sheet's
particles, keypoint policy), ../cloud_obs.py (point_cloud -> the network's input).
"""
