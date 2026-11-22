"""Watch a fold policy live in a browser (viser), on one of the evaluation targets.

    experiments/viewer.sh vbd                                     # Newton VBD cloth, port 8080
    experiments/viewer.sh rigidbody_flat --policy "<menu name>"   # PhysX rigid sheet, port 8081
    experiments/viewer.sh rigidbody_folded --policy outputs/distill/<run>/student_last.pth

One viewer runs one target (``targets.py``), built as ``scripts/evaluation/eval.py`` builds it: the rigid
sheet in the starting policy's training world (its recipe), domain randomisation on (``--start rand``)
or off with the start pinned (``--start pinned``, VBD's default); a point-cloud student reads the env's
cloud. On the rigid sheet every env carries all seven shapes and the sheet sizes run evenly from
env 0 (smallest) to the last env. The policy menu is ``policies.yaml`` (or ``--policies``).

Modules: ``app`` (command line, the env), ``targets``, ``policies`` (menu, players), ``scene`` (bodies'
geometry and poses, Newton / PhysX), ``viewer`` (state, panels, status, loop), ``feature`` (the
interface of the optional parts below), ``fold_pool`` (shape per env, even sizes), ``overlays``
(keypoints, goal ghost), ``vbd`` (cloth sheet, fingers through it), ``point_cloud`` (the student's cloud,
camera noise).
"""
