"""GPU detection for NVIDIA (nvidia-smi), AMD (rocm-smi), Apple Silicon and CPU-only machines.

Every probe is best effort: a missing tool or an unexpected output format gives an empty
list, never an exception, so the tools keep working on any machine.
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys


def _run(*argv: str, timeout: float = 20) -> str | None:
    if not shutil.which(argv[0]):
        return None
    try:
        out = subprocess.run(argv, capture_output=True, text=True, timeout=timeout,
                             encoding="utf-8", errors="replace")
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def _num(x, default: float | None = None) -> float | None:
    try:
        return float(str(x).strip().split()[0])
    except (ValueError, IndexError):
        return default


def nvidia() -> tuple[list[dict], list[dict]]:
    fields = ["index", "name", "utilization.gpu", "memory.used", "memory.total",
              "temperature.gpu", "power.draw", "power.limit", "clocks.sm",
              "clocks.max.sm", "pstate", "driver_version"]
    out = _run("nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits")
    if not out:
        return [], []
    gpus = []
    for line in out.strip().splitlines():
        v = [x.strip() for x in line.split(",")]
        if len(v) < len(fields):
            continue
        mem = _num(v[3], 0.0)
        power, limit = _num(v[6]), _num(v[7])
        pct = round(100 * power / limit, 1) if power is not None and limit else None
        gpus.append({
            "vendor": "nvidia", "index": int(_num(v[0], 0)), "name": v[1],
            "utilization_pct_reported": _num(v[2]),
            "memory_used_mib": int(mem), "memory_total_mib": int(_num(v[4], 0.0)),
            "temperature_c": _num(v[5]), "power_draw_w": power, "power_limit_w": limit,
            "power_pct": pct, "sm_clock_mhz": _num(v[8]), "sm_clock_max_mhz": _num(v[9]),
            "pstate": v[10], "driver": v[11],
            "busy": (pct or 0) > 25 or mem > 1000,
        })
    procs_out = _run("nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                     "--format=csv,noheader,nounits") or ""
    procs = [dict(zip(["pid", "process", "memory_mib"], [x.strip() for x in l.split(",")]))
             for l in procs_out.strip().splitlines() if l.strip()]
    return gpus, procs


def amd() -> list[dict]:
    out = _run("rocm-smi", "--showproductname", "--showmeminfo", "vram", "--showpower",
               "--showtemp", "--showuse", "--json")
    if not out:
        return []
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        return []

    def pick(d: dict, *needles: str):
        for k, v in d.items():
            if all(n.lower() in k.lower() for n in needles):
                return v
        return None

    gpus = []
    for i, (card, d) in enumerate(sorted((k, v) for k, v in data.items() if k.startswith("card"))):
        used_b, total_b = _num(pick(d, "vram", "used")), _num(pick(d, "vram", "total memory"))
        util, power = _num(pick(d, "gpu use")), _num(pick(d, "power"))
        gpus.append({
            "vendor": "amd", "index": i,
            "name": pick(d, "card series") or pick(d, "product name") or card,
            "utilization_pct_reported": util,
            "memory_used_mib": int(used_b / 2**20) if used_b is not None else None,
            "memory_total_mib": int(total_b / 2**20) if total_b is not None else None,
            "temperature_c": _num(pick(d, "temperature")), "power_draw_w": power,
            "busy": (util or 0) > 10 or (used_b or 0) > 1000 * 2**20,
        })
    return gpus


def apple() -> list[dict]:
    if sys.platform != "darwin" or platform.machine() != "arm64":
        return []
    name, cores = "Apple Silicon GPU", None
    out = _run("system_profiler", "SPDisplaysDataType", "-json")
    try:
        d = json.loads(out)["SPDisplaysDataType"][0] if out else {}
        name, cores = d.get("sppci_model", name), _num(d.get("sppci_cores"))
    except (json.JSONDecodeError, KeyError, IndexError):
        pass
    import psutil
    return [{"vendor": "apple", "index": 0, "name": name,
             "gpu_cores": int(cores) if cores else None,
             "memory_total_mib": int(psutil.virtual_memory().total / 2**20),
             "memory_note": "unified memory shared with the CPU",
             "torch_device": "mps"}]


def status() -> dict:
    gpus, procs = nvidia()
    gpus = gpus or amd() or apple()
    result = {"gpus": gpus, "processes": procs}
    if not gpus:
        result["note"] = ("No GPU detected (no nvidia-smi, rocm-smi or Apple Silicon). "
                          "Everything still works; compute runs on the CPU.")
    return result


def memory_used_mib() -> list[int | None]:
    return [g.get("memory_used_mib") for g in status()["gpus"]]
