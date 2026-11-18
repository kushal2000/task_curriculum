"""Isaac Lab 3.0 / Newton support layer for the VBD backend.

The Play task code (``isaacsimenvs/tasks/play/``) is written against Isaac Lab 2.x and is shared
verbatim with the PhysX backend. Everything needed to make it load and behave correctly under Isaac
Lab 3.0 + Newton/MJWarp lives here, so the two backends run the same reward, observation and action
code and a difference between them is attributable to physics.

    compat              import-time relocations (3.0 moved symbols the task code imports by 2.x paths)
    patches             runtime patches -- quaternion conventions, actuator fixes, cloning, Newton
                        solver setup. Vendored from github.com/kushal2000/isaac_newton @ beb9efb.
    physx_schema_shim   the part of pxr.PhysxSchema the kit-less stack lacks
    usd_cache           URDF -> USD conversions baked under Isaac Sim, read by the kit-less stack
"""
