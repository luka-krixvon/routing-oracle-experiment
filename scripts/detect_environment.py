"""Detect the full hardware / NVIDIA / Python-ML / runtime environment and emit a
reproducibility report for the paper's Experimental Setup.

Outputs (reports/environment/):
  environment_report.json     full machine-readable record
  environment_report.md       human-readable (safe to commit / share)
  paper_environment_summary.md formal English paragraph for the paper, versions auto-filled

Design: stdlib-first; every third-party probe is guarded (missing tool/lib -> null, never
crashes). Secrets are masked (tokens reported as present-only; env var values whitelisted;
home dir collapsed to ~). Run BEFORE experiments; re-run after to log final GPU/disk state.

  python scripts/detect_environment.py            # threshold 30 GB
  python scripts/detect_environment.py --min-free-gb 25 --anonymize
"""
from __future__ import annotations
import argparse, json, os, platform, re, shutil, socket, subprocess, sys
from pathlib import Path

HOME = str(Path.home())
SENSITIVE = re.compile(r"(TOKEN|KEY|SECRET|PASSWORD|PASSWD|AUTH|CREDENTIAL)", re.I)
ENV_WHITELIST = ["CUDA_VISIBLE_DEVICES", "HF_HOME", "TRANSFORMERS_CACHE", "HF_HUB_CACHE",
                 "CONDA_DEFAULT_ENV", "CONDA_PREFIX", "VLLM_WORKER_MULTIPROC_METHOD"]


def sh(cmd, timeout=20):
    """Run a command; return stripped stdout on success, else None (never raises,
    never returns stderr — so a missing tool yields a clean null, not an error string)."""
    try:
        out = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True,
                             text=True, timeout=timeout)
        s = out.stdout.strip()
        return s if (out.returncode == 0 and s) else None
    except Exception:
        return None


def ver(pkg):
    try:
        from importlib.metadata import version
        return version(pkg)
    except Exception:
        return None


def mask_path(p, anonymize=False):
    if p is None:
        return None
    p = str(p).replace(HOME, "~")
    if anonymize:
        p = re.sub(r"/(home|Users)/[^/]+", r"/\1/<user>", p)
    return p


def human(n):
    if n is None:
        return None
    for u in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} PB"


def dir_size(path):
    out = sh(["du", "-sb", str(path)])
    if out:
        try:
            return int(out.split()[0])
        except Exception:
            pass
    total = 0
    for root, _, files in os.walk(path):
        if ".git" in root:
            continue
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


# --------------------------------------------------------------------------- A
def hardware(anon):
    cpu = (sh("grep -m1 'model name' /proc/cpuinfo | cut -d: -f2") or
           sh("sysctl -n machdep.cpu.brand_string") or platform.processor() or None)
    try:
        import psutil
        ram = psutil.virtual_memory().total
        phys = psutil.cpu_count(logical=False)
    except Exception:
        ram, phys = None, None
        mt = sh("grep MemTotal /proc/meminfo")
        if mt:
            try:
                ram = int(re.search(r"(\d+)", mt).group(1)) * 1024
            except Exception:
                pass
        if ram is None:
            ms = sh("sysctl -n hw.memsize")
            ram = int(ms) if ms and ms.isdigit() else None
    du = shutil.disk_usage(os.getcwd())
    return {
        "hostname": "<host>" if anon else socket.gethostname(),
        "os": platform.platform(),
        "os_release": (sh("grep -m1 PRETTY_NAME /etc/os-release") or "").split("=")[-1].strip('"') or None,
        "cpu_model": (cpu or "").strip() or None,
        "cpu_cores_physical": phys,
        "cpu_threads_logical": os.cpu_count(),
        "ram_total": human(ram),
        "disk_total": human(du.total), "disk_used": human(du.used), "disk_free": human(du.free),
        "disk_free_gb": round(du.free / 1e9, 1),
        "project_dir_size": human(dir_size(os.getcwd())),
        "hf_cache_size": _hf_cache_size(),
        "gpus": _gpus(),
    }


def _hf_cache_size():
    try:
        from huggingface_hub import scan_cache_dir
        return scan_cache_dir().size_on_disk_str
    except Exception:
        hh = os.environ.get("HF_HOME", os.path.join(HOME, ".cache", "huggingface"))
        out = sh(["du", "-sh", hh])
        return out.split()[0] if out else None


def _gpus():
    q = ("index,name,memory.total,memory.used,temperature.gpu,power.draw,pci.bus_id,"
         "compute_cap,driver_version")
    out = sh(f"nvidia-smi --query-gpu={q} --format=csv,noheader")
    gpus = []
    if out:
        for line in out.splitlines():
            c = [x.strip() for x in line.split(",")]
            if len(c) >= 9:
                gpus.append({"index": c[0], "name": c[1], "memory_total": c[2],
                             "memory_used": c[3], "temperature_C": c[4], "power": c[5],
                             "pci_bus_id": c[6], "compute_capability": c[7]})
        return gpus
    # fallback: torch
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                gpus.append({"index": i, "name": p.name,
                             "memory_total": human(p.total_memory),
                             "compute_capability": f"{p.major}.{p.minor}"})
    except Exception:
        pass
    return gpus


# --------------------------------------------------------------------------- B
def nvidia():
    info = {"driver_version": None, "cuda_toolkit": None, "cuda_runtime": None,
            "cuda_available": None, "cudnn_version": None, "nccl_version": None,
            "torch_detects_cuda": None, "torch_cuda_build": None,
            "cudnn_backend_enabled": None, "device_names": [], "compute_capabilities": [],
            "nvidia_smi_summary": None}
    info["driver_version"] = sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1")
    nvcc = sh("nvcc --version")
    if nvcc:
        m = re.search(r"release\s+([\d.]+)", nvcc)
        info["cuda_toolkit"] = m.group(1) if m else None
    smi = sh("nvidia-smi")
    if smi:
        info["nvidia_smi_summary"] = "\n".join(smi.splitlines()[:12])
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        info["torch_detects_cuda"] = torch.cuda.is_available()
        info["torch_cuda_build"] = torch.version.cuda
        info["cuda_runtime"] = torch.version.cuda
        try:
            info["cudnn_version"] = torch.backends.cudnn.version()
            info["cudnn_backend_enabled"] = bool(torch.backends.cudnn.enabled)
        except Exception:
            pass
        try:
            n = torch.cuda.nccl.version()
            info["nccl_version"] = ".".join(map(str, n)) if isinstance(n, (tuple, list)) else str(n)
        except Exception:
            pass
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                info["device_names"].append(torch.cuda.get_device_name(i))
                cc = torch.cuda.get_device_capability(i)
                info["compute_capabilities"].append(f"{cc[0]}.{cc[1]}")
    except Exception:
        pass
    return info


# --------------------------------------------------------------------------- C
def python_ml():
    pkgs = ["torch", "torchvision", "torchaudio", "transformers", "accelerate", "datasets",
            "huggingface_hub", "tokenizers", "safetensors", "sentencepiece", "bitsandbytes",
            "vllm", "flash-attn", "flash_attn", "xformers", "numpy", "pandas", "scipy",
            "scikit-learn", "matplotlib", "seaborn", "statsmodels"]
    versions = {p: ver(p) for p in pkgs}
    return {
        "python_version": platform.python_version(),
        "python_executable": mask_path(sys.executable),
        "pip_version": (ver("pip")),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "packages": {k: v for k, v in versions.items() if v is not None},
        "packages_missing": [k for k, v in versions.items() if v is None],
    }


# --------------------------------------------------------------------------- D
def runtime(min_free_gb, anon):
    g = lambda c: sh(c)
    token_present = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
                        or os.path.exists(os.path.join(HOME, ".cache", "huggingface", "token")))
    env = {}
    for k in ENV_WHITELIST:
        if k in os.environ:
            env[k] = mask_path(os.environ[k], anon)
    for k in os.environ:
        if SENSITIVE.search(k):
            env[k] = "**** (masked)"
    du = shutil.disk_usage(os.getcwd())
    return {
        "cwd": mask_path(os.getcwd(), anon),
        "git_remote": g("git config --get remote.origin.url"),
        "git_branch": g("git rev-parse --abbrev-ref HEAD"),
        "git_commit": g("git rev-parse HEAD"),
        "git_uncommitted_changes": bool(g("git status --porcelain")),
        "gitignore_present": os.path.exists(".gitignore"),
        "dirs_present": {d: os.path.isdir(d) for d in ["reports", "results", "logs", "data",
                                                       "paper/figs", "configs", "scripts", "src"]},
        "hf_cache_path": mask_path(os.environ.get("HF_HOME",
                                   os.path.join(HOME, ".cache", "huggingface")), anon),
        "env_vars": env,
        "hf_token_present": "present (masked)" if token_present else "no",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "(unset = all)"),
        "disk_free_gb": round(du.free / 1e9, 1),
        "disk_enough_to_start": du.free / 1e9 >= min_free_gb,
        "min_free_gb_threshold": min_free_gb,
    }


# --------------------------------------------------------------------------- emit
def write_md(rep):
    H = rep["hardware"]; N = rep["nvidia"]; P = rep["python_ml"]; R = rep["runtime"]
    L = ["# Environment Report", f"_generated: {rep['generated_utc']}_  ·  detect_environment.py\n"]
    L += ["## A. Hardware"]
    for k in ["hostname", "os", "os_release", "cpu_model", "cpu_cores_physical",
              "cpu_threads_logical", "ram_total", "disk_total", "disk_used", "disk_free",
              "project_dir_size", "hf_cache_size"]:
        L.append(f"- **{k}**: {H.get(k)}")
    L.append("\n### GPUs")
    if H["gpus"]:
        L.append("| idx | name | VRAM total | used | temp | power | compute |")
        L.append("|---|---|---|---|---|---|---|")
        for g in H["gpus"]:
            L.append(f"| {g.get('index')} | {g.get('name')} | {g.get('memory_total')} | "
                     f"{g.get('memory_used')} | {g.get('temperature_C')} | {g.get('power')} | "
                     f"{g.get('compute_capability')} |")
    else:
        L.append("_no GPU detected_")
    L += ["\n## B. NVIDIA Environment"]
    for k in ["driver_version", "cuda_toolkit", "cuda_runtime", "cuda_available",
              "cudnn_version", "nccl_version", "torch_detects_cuda", "torch_cuda_build",
              "cudnn_backend_enabled", "device_names", "compute_capabilities"]:
        L.append(f"- **{k}**: {N.get(k)}")
    if N.get("nvidia_smi_summary"):
        L += ["\n<details><summary>nvidia-smi</summary>\n\n```", N["nvidia_smi_summary"], "```\n</details>"]
    L += ["\n## C. Python / ML Software", f"- **python**: {P['python_version']}  ·  **pip**: {P['pip_version']}"
          f"  ·  **conda_env**: {P['conda_env']}", "\n| package | version |", "|---|---|"]
    for k, v in P["packages"].items():
        L.append(f"| {k} | {v} |")
    L.append(f"\n_not installed_: {', '.join(P['packages_missing']) or 'none'}")
    L += ["\n## D. Runtime"]
    for k in ["cwd", "git_remote", "git_branch", "git_commit", "git_uncommitted_changes",
              "gitignore_present", "hf_cache_path", "hf_token_present",
              "cuda_visible_devices", "disk_free_gb", "disk_enough_to_start"]:
        L.append(f"- **{k}**: {R.get(k)}")
    L.append(f"- **dirs_present**: {R['dirs_present']}")
    L.append(f"- **env_vars (whitelisted/masked)**: {R['env_vars']}")
    return "\n".join(L) + "\n"


def write_paper(rep):
    H = rep["hardware"]; N = rep["nvidia"]; P = rep["python_ml"]; pk = P["packages"]
    gpus = H["gpus"]; ng = len(gpus)
    gname = gpus[0]["name"] if gpus else "N/A"
    vram = gpus[0].get("memory_total", "N/A") if gpus else "N/A"
    cc = gpus[0].get("compute_capability") if gpus else (N["compute_capabilities"][0]
         if N["compute_capabilities"] else "N/A")
    gpu_clause = (f"{ng}\\times{{}} {gname} GPUs ({vram} each)" if ng != 1
                  else f"a single {gname} GPU ({vram})") if gpus else "an NVIDIA GPU"

    def f(x, d="N/A"):
        return x if x else d
    para = (
        f"All experiments were conducted on a GPU server ({f(H.get('os_release') or H.get('os'))}, "
        f"{f(H.get('cpu_model'))}, {f(H.get('ram_total'))} RAM) equipped with {gpu_clause}, "
        f"compute capability {f(cc)}. The NVIDIA software stack comprised driver version "
        f"{f(N.get('driver_version'))}, CUDA toolkit {f(N.get('cuda_toolkit'))} "
        f"(runtime {f(N.get('cuda_runtime'))}), cuDNN {f(N.get('cudnn_version'))}, and NCCL "
        f"{f(N.get('nccl_version'))}. Models were served with vLLM {f(pk.get('vllm'))} on "
        f"PyTorch {f(pk.get('torch'))} (CUDA build {f(N.get('torch_cuda_build'))}, "
        f"cuDNN backend {'enabled' if N.get('cudnn_backend_enabled') else 'N/A'}), using "
        f"Transformers {f(pk.get('transformers'))}, Accelerate {f(pk.get('accelerate'))}, "
        f"Tokenizers {f(pk.get('tokenizers'))}, and Datasets {f(pk.get('datasets'))}, under "
        f"Python {f(P.get('python_version'))}. The complete environment — including per-GPU "
        f"memory, CUDA runtime, NCCL, and exact package versions — was captured automatically "
        f"by a detection script and is released with the code to support reproducibility.")
    spec = ["| Component | Specification |", "|---|---|",
            f"| GPU | {ng}× {gname} ({vram}) |" if gpus else "| GPU | N/A |",
            f"| NVIDIA driver | {f(N.get('driver_version'))} |",
            f"| CUDA | toolkit {f(N.get('cuda_toolkit'))} / runtime {f(N.get('cuda_runtime'))} |",
            f"| cuDNN / NCCL | {f(N.get('cudnn_version'))} / {f(N.get('nccl_version'))} |",
            f"| PyTorch | {f(pk.get('torch'))} (CUDA {f(N.get('torch_cuda_build'))}) |",
            f"| vLLM / Transformers | {f(pk.get('vllm'))} / {f(pk.get('transformers'))} |",
            f"| Python | {f(P.get('python_version'))} |",
            f"| CPU / RAM | {f(H.get('cpu_model'))} / {f(H.get('ram_total'))} |"]
    return ("# Experimental Setup — environment (auto-generated, paper-ready)\n\n"
            "## Suggested paragraph\n\n" + para + "\n\n## Spec table\n\n" + "\n".join(spec) +
            "\n\n_Reproducibility: full record in environment_report.json; "
            "regenerate via `python scripts/detect_environment.py`._\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-free-gb", type=float, default=30)
    ap.add_argument("--anonymize", action="store_true", help="mask hostname + usernames in paths")
    ap.add_argument("--outdir", default=None)
    a = ap.parse_args()
    root = Path(__file__).resolve().parent.parent
    outdir = Path(a.outdir) if a.outdir else root / "reports" / "environment"
    outdir.mkdir(parents=True, exist_ok=True)
    import datetime
    rep = {"generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
           "hardware": hardware(a.anonymize), "nvidia": nvidia(),
           "python_ml": python_ml(), "runtime": runtime(a.min_free_gb, a.anonymize)}
    (outdir / "environment_report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    (outdir / "environment_report.md").write_text(write_md(rep))
    (outdir / "paper_environment_summary.md").write_text(write_paper(rep))
    g = rep["hardware"]["gpus"]; r = rep["runtime"]
    print(f"environment report -> {outdir}/")
    print(f"  GPUs: {len(g)} ({', '.join(x.get('name','?') for x in g) or 'none detected'})")
    print(f"  disk free: {r['disk_free_gb']} GB  enough_to_start(>= {a.min_free_gb}): {r['disk_enough_to_start']}")
    print(f"  HF token: {r['hf_token_present']}")
    if not r["disk_enough_to_start"]:
        print("  !! disk below threshold — free space before downloading models")


if __name__ == "__main__":
    main()
