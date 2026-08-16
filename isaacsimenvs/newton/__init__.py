"""Isaac Lab 3.0 / Newton support layer for the Play task.

The task code in ``isaacsimenvs/tasks/play/`` is written against Isaac Lab 2.x and is shared
verbatim with the PhysX env. Everything needed to make it load and behave correctly under Isaac
Lab 3.0 + Newton/MJWarp lives here instead, so the two backends provably run the same reward,
observation and action code and a difference between them is attributable to physics.

    compat    import-time relocations (3.0 moved symbols the task code imports by 2.x paths)
    patches   runtime patches -- quaternion conventions, actuator fixes, cloning, Newton solver
              setup. Vendored from github.com/kushal2000/isaac_newton @ beb9efb.

Vendored rather than depended on: that repository is an investigation log with a moving HEAD and
a history of retractions, so pinning a copy is safer than tracking it. Its four-backend
``Backend`` abstraction is not carried over -- here the registered task id
(``Isaacsimenvs-PlayNewton-Direct-v0``) *is* the backend switch, and construction lives in the
env class rather than in a separate builder.
"""
