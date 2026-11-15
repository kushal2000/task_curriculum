# Vendored third-party source

Plain copies of upstream repos at pinned commits, committed so a clone of this repo is
complete. Do not edit them in place: change the pin in `third_party/vendor.sh` and
rerun it. Each directory keeps its upstream LICENSE. Upstream's git-lfs files are left out:
without their LFS objects they are only pointer stubs (IsaacLab: test golden images under
`source/*/test/`).

| dir | upstream | commit | used by |
|---|---|---|---|
| `IsaacLab/` | https://github.com/isaac-sim/IsaacLab | `a27ec0beb2345cb4618e31b8e42bc09c557ee193` | .venv/isaaclab3 (Newton / VBD) |
| `aioflexiv/` | https://github.com/younghyopark/aioflexiv | `adce58442f945ec74c684b78f228c6aa1d4cb8f8` | deploy/ (Flexiv arm) |
| `aiowuji/` | https://github.com/Halo123188/aiowuji | `f05afcc6fd536f4dfa7f3e6a57f928265e1d5aab` | deploy/ (Wuji hand) |
