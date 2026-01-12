import os
import json

import argparse
import yaml

from datetime import datetime
from huggingface_hub import snapshot_download
from pathlib import Path
import shutil
from sentence_transformers import CrossEncoder
from sentence_transformers import SentenceTransformer

def read_fingerprint(model_dir: Path) -> dict | None:
    fp = model_dir / ".model_fingerprint.json"
    if not fp.exists():
        return None
    try:
        return json.load(open(fp))
    except Exception:
        return None


def write_fingerprint(
    model_dir: Path,
    *,
    hf_id: str,
    revision: str,
    model_type: str,
):
    data = {
        "hf_id": hf_id,
        "revision": revision,
        "model_type": model_type,
        "written_at": datetime.utcnow().isoformat() + "Z",
    }
    with open(model_dir / ".model_fingerprint.json", "w") as f:
        json.dump(data, f, indent=2)

def print_folder_structure(folder_path, prefix=""):
    """
    Recursively print the folder structure of folder_path
    """
    folder_path = Path(folder_path)
    if not folder_path.exists():
        print(f"{folder_path} does not exist!", flush=True)
        return

    for item in sorted(folder_path.iterdir()):
        if item.is_dir():
            print(f"{prefix}[DIR]  {item.name}", flush=True)
            print_folder_structure(item, prefix + "    ")
        else:
            print(f"{prefix}[FILE] {item.name}", flush=True)


def remove_safetensors(target_path: Path):
    """Remove all safetensors files (self-healing)"""
    for f in target_path.glob("*.safetensors*"):
        print(f"[REMOVE] {f.name}", flush=True)
        f.unlink()

def flatten_hf_snapshot(snapshot_path, target_path, allow_safetensors: bool):
    """
    Copy only essential files from HF snapshot to target_path.
    This ensures runtime PVC and S3 only contain required files.
    """
    Path(target_path).mkdir(parents=True, exist_ok=True)

    for src_file in snapshot_path.rglob("*"):
        if not src_file.is_file():
            continue

        rel = src_file.relative_to(snapshot_path)

        # 🚫 Block safetensors if not allowed
        if not allow_safetensors and src_file.suffix == ".safetensors":
            print(f"[SKIP] {rel} (safetensors disabled)", flush=True)
            continue

        dst = target_path / rel
        if dst.exists():
            print(f"[SKIP] {rel} already exists", flush=True)
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src_file, dst)
        print(f"[COPIED] {rel}", flush=True)

    print(f"Model snapshot flattened to {target_path}", flush=True)


def download_and_deploy_models(config_file: str, meta_file: str, pvc_base_path: str):
    with open(config_file) as f:
        models_config = yaml.safe_load(f)

    if os.path.exists(meta_file):
        with open(meta_file) as f:
            meta = json.load(f)
    else:
        meta = {}

    for model_key, cfg in models_config.items():
        if "model" not in model_key:
            continue

        label = cfg["label"]
        hf_id = cfg["hf_id"]
        model_type = cfg.get("type", "causal")

        # --------------------------------------------------
        # Resolve PVC path
        # --------------------------------------------------
        local_override = cfg.get("local_path")

        if local_override:
            # Always resolve under pvc_base_path in CI or local runs
            pvc_path = Path(pvc_base_path) / Path(local_override).name
        else:
            pvc_path = Path(pvc_base_path) / model_key

        pvc_path.mkdir(parents=True, exist_ok=True)

        revision = cfg.get("revision", "main")
        fingerprint = read_fingerprint(pvc_path)

        # --------------------------------------------------
        # Decide reuse vs wipe (PVC is the authority)
        # --------------------------------------------------
        if fingerprint:
            if (
                    fingerprint.get("hf_id") == hf_id
                    and fingerprint.get("revision") == revision
                    and fingerprint.get("model_type") == model_type
            ):
                print(f"[SKIP] {model_key} already installed ({hf_id})", flush=True)
                continue
            else:
                print(
                    f"[MODEL CHANGE] {model_key}: "
                    f"{fingerprint.get('hf_id')} → {hf_id}",
                    flush=True,
                )
                shutil.rmtree(pvc_path, ignore_errors=True)
                pvc_path.mkdir(parents=True, exist_ok=True)
        else:
            # No fingerprint = unsafe / partial state
            if any(pvc_path.iterdir()):
                print(f"[NO FINGERPRINT] wiping {model_key}", flush=True)
                shutil.rmtree(pvc_path, ignore_errors=True)
                pvc_path.mkdir(parents=True, exist_ok=True)
        # --------------------------------------------------
        # Download HF snapshot (cached)
        # --------------------------------------------------
        snapshot_path = Path(
            snapshot_download(repo_id=hf_id, revision=cfg.get("revision", "main"))
        )

        print(f"[HF] Snapshot ready at {snapshot_path}", flush=True)

        # --------------------------------------------------
        # Materialize model
        # --------------------------------------------------
        if model_type == "embedding":
            model_name = hf_id.split("/")[-1]
            target = pvc_path / "sentence-transformers" / model_name

            if not target.exists():
                model = SentenceTransformer(hf_id, cache_folder=str(pvc_path / "cache"))
                model.save(str(target))
                print(f"[EMBEDDING] Saved to {target}", flush=True)

        elif model_type == "cross-encoder":
            model_name = hf_id.split("/")[-1]
            target = pvc_path / "cross-encoder" / model_name

            if not target.exists():
                model = CrossEncoder(hf_id, device="cpu", trust_remote_code=True)
                model.save(str(target))
                print(f"[CROSS-ENCODER] Saved to {target}", flush=True)

        else:
            # generator / evaluator / other LLMs
            flatten_hf_snapshot(
                snapshot_path,
                pvc_path,
                allow_safetensors=True,
            )

        # --------------------------------------------------
        # Update metadata
        # --------------------------------------------------
        meta[model_key] = {
            "label": label,
            "hf_id": hf_id,
            "pvc_path": str(pvc_path),
            "downloaded_at": datetime.utcnow().isoformat(),
        }

        write_fingerprint(
            pvc_path,
            hf_id=hf_id,
            revision=revision,
            model_type=model_type,
        )

    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)

    print("✅ All models processed successfully", flush=True)


if __name__ == "__main__":

    import glob

    #for lock_dir in glob.glob("/models/hf/**/*.lock", recursive=True):
    #    print(f"Removing stale HF lock: {lock_dir}", flush=True)
    #    shutil.rmtree(lock_dir, ignore_errors=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to model config YAML")
    parser.add_argument("--meta", required=True, help="Path to metadata JSON")
    parser.add_argument("--pvc_base", required=True, help="Local path to store models")

    args = parser.parse_args()

    hf_base = os.getenv("HF_HOME", str(Path(args.pvc_base) / "hf_cache"))

    for lock_dir in Path(hf_base).rglob("*.lock"):
        print(f"Removing stale HF lock: {lock_dir}", flush=True)
        shutil.rmtree(lock_dir, ignore_errors=True)




    download_and_deploy_models(args.config, args.meta, args.pvc_base)
