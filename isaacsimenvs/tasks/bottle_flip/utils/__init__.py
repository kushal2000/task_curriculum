"""Stateless helpers for bottle flipping. Every function takes `env` first.

`generate_bottle` is free of torch and Isaac imports so it can be unit-tested without
booting Kit; its inertia model is shared with the runtime fill-level setter in
`reset_utils`.
"""
