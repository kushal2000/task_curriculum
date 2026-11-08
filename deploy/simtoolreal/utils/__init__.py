"""Helpers for the deploy processes (the nodes live one folder up). numpy only.

config       config.yaml + command-line overrides
link         ZMQ messages between the processes (the topic table is there)
quat         quaternion helpers (wxyz)
spec         the training env's facts (spec json) + robot base <-> env frame
robot_model  forward kinematics: palm, fingertips
observation  the policy's 134-number observation
goals        sample goals, test whether the object reached one
fake_hand    a stand-in for the Wuji hand driver when no hand is connected (robot.hand=fake)
"""
