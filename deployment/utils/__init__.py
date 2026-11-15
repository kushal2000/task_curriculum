"""Code shared by every deploy task (simtoolreal, rigid2deform, ...). numpy only, except config (omegaconf)
and link (pyzmq). The shared nodes are in deploy/nodes/, each task's run.py / policy.py in its folder.

config       a task's config.yaml + command-line overrides
link         ZMQ messages between the processes (the topic table is there), starting nodes
quat         quaternion helpers (wxyz)
spec         the training env's facts (spec json) + robot base <-> env frame
spec_export  building a spec json from a training branch, in git (the tasks' checks/export_*spec.py)
robot_model  forward kinematics: palm, fingertips
observation  the policy's 134-number observation; the success keypoint distance
policy_net   the trained network, action -> joint targets, waiting for the robot and the object
runner       a task's run.py: start the nodes, watch them, stop them
fake_hand    a stand-in for the Wuji hand driver when no hand is connected (robot.hand=fake)
openloop     the sim rollout robot.mode=openloop sends (simtoolreal/checks/record_lift.py), where to put the object
video        H.264 mp4 files (the recorder's camera video, playback's export)
"""
