"""Write outputs/cloth_bench/<tag>_n<envs>.json from the THROUGHPUT lines of clothbench slurm logs.

    python experiments/nov16_vbd_cloth/bench_from_log.py slurm_logs/clothbench_<job>.out ...
"""
import json
import re
import sys
from pathlib import Path

out = Path(__file__).resolve().parents[2] / "outputs/cloth_bench"
out.mkdir(parents=True, exist_ok=True)
for log in sys.argv[1:]:
    tag = None
    for line in Path(log).read_text().splitlines():
        m = re.match(r"\[bench\] (\S+) n=(\d+)", line)
        if m:
            tag = m.group(1)
        elif line.startswith("THROUGHPUT ") and tag:
            d = json.loads(line[len("THROUGHPUT "):])
            p = out / f"{tag}_n{d['num_envs']}.json"
            p.write_text(json.dumps(d, indent=1))
            print(p.name, d["env_steps_per_s"], d["gpu_mem_gb"])
