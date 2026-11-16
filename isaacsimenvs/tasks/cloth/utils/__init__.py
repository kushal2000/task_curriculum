"""The cloth task as plain functions, shared by the PhysX and the Newton env.

Every function takes the env (for its config and buffers) and/or a :class:`state.ClothState`; none
touches a simulator object, so a backend only has to fill a ``ClothState`` and apply the writes.
Nothing here imports Isaac Lab except ``cfg.py`` (configclasses), so the tests run without Kit.
"""
