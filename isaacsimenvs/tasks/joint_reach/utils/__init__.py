"""Stateless helpers for joint reach. Every function takes `env` first.

    scene_utils        table geometry, robot points, workspace test, parking the object, PhysX sizing
    goal_bank          offline goal generation: collision-checked bank and random-walk chains, cached
    goal_utils         per-episode goal stepping along the chains and goal-segment tallies
    reset_utils        buffers, robot placement, per-episode reset and gain randomisation
    reward_utils       reward terms
    termination_utils  goal error and timeouts
    obs_utils          policy and critic observations
    logging_utils      extras for EnvStatsAlgoObserver
"""
