"""Isaac Lab / Newton compatibility required by the cloth VBD backend.

``compat`` handles import-time API differences; ``patches`` supplies scene and
actuator fixes; ``physx_schema_shim`` authors USD attributes without Kit; and
``usd_cache`` reads assets prepared by ``isaacsimenvs.scripts.assets.prepare_usd_cache``.

Task rewards, observations, and actions live in cloth's MDP components.
"""
