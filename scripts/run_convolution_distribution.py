"""Run only the authorized convolution-distribution study, independently report, then stop."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import yaml

ROOT = Path(__file__).resolve().parents[1]


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def verified_ready(config):
    root = Path(config["output_dir"])/"complete"
    required = ["summary.json", "study_manifest.json", "REPORT.md", "verification_manifest.json"]
    if not all((root/p).is_file() for p in required): return False
    s = json.loads((root/"summary.json").read_text()); scope = json.loads((root/"study_manifest.json").read_text())
    groups = config["outer_splits"]*len(config["projection_seeds"])
    fits = groups*len(config["arms"])*(config["inner_splits"]*len(config["ridge_alphas"])+1)
    contexts = groups*(config["inner_splits"]+1)
    source_fits=groups*(config['inner_splits']*len(config['ridge_alphas'])+1)
    return (s.get('complete') is True and s.get('study_fingerprint')==scope.get('study_fingerprint')
            and scope.get('config')==config and s.get('fits')==s.get('all_models_restored')==fits
            and s.get('new_models_restored')==s.get('new_fits')==3*source_fits
            and s.get('source_models_restored')==s.get('source_ridge_contexts')==source_fits
            and s.get('outer_models_restored')==groups*len(config['arms'])
            and s.get('feature_contexts_reconstructed')==s.get('matched_context_checks')==contexts
            and all(s.get(k) is True for k in ('all_original_beats_replayed','original_cohort_unchanged','source_files_unchanged','baseline_probabilities_exact'))
            and all(Path(k).is_file() and sha(Path(k))==v for k,v in scope.get('code',{}).items())
            and all(Path(k).is_file() and sha(Path(k))==v for k,v in scope.get('source',{}).get('protected_files',{}).items())
            and s.get('report_sha256')==sha(root/'REPORT.md')
            and s.get('verification_manifest_sha256')==sha(root/'verification_manifest.json'))



def publish(config):
    if not verified_ready(config): raise RuntimeError("Publication requires independently verified complete coverage")
    if subprocess.check_output(["git", "branch", "--show-current"], text=True).strip() != "main": raise RuntimeError("Publication requires main")
    if subprocess.check_output(["git", "diff", "--cached", "--name-only"], text=True).strip(): raise RuntimeError("Unrelated staged changes; refusing to include them")
    # Stage only aggregate outputs. Pre-existing dirty files in other studies stay untouched.
    root = Path(config["output_dir"])
    files = [root/"source_preflight/summary.json", root/"benchmark.json", root/"smoke/summary.json"]+[root/"complete"/n for n in ("summary.json", "REPORT.md", "study_manifest.json", "verification_manifest.json")]
    files += sorted((root/"complete").glob("confusion_patient_*.csv"))+sorted((root/"complete").glob("confusion_patient_*.png"))
    subprocess.run(["git", "add", "--", *map(str, files)], check=True)
    if subprocess.run(["git", "diff", "--cached", "--quiet"]).returncode:
        subprocess.run(["git", "commit", "-m", "Publish independently replayed intact-base convolution distributions"], check=True)
    for attempt in range(3):
        if subprocess.run(["git", "push", "origin", "HEAD:main"]).returncode == 0: return
        if attempt < 2: time.sleep(10)
    raise RuntimeError("Verified result committed locally; GitHub push failed")


def main():
    from run_residual_context import acquire_lock
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--config", default="configs/convolution_distribution_v1.yaml"); p.add_argument("--publish", action="store_true")
    args = p.parse_args(); os.chdir(ROOT); config = yaml.safe_load(Path(args.config).read_text())
    # The lock path belongs to this output namespace, including on Windows.
    lock = acquire_lock(Path(config["output_dir"])/"runner.lock")
    progress = Path(config["output_dir"])/"runner.json"
    def save(state, **kw):
        value = dict(pid=os.getpid(), updated_utc=datetime.now(timezone.utc).isoformat(), state=state, **kw)
        temporary = progress.with_suffix(".partial.json"); temporary.write_text(json.dumps(value, indent=2)); temporary.replace(progress)
    verified = False
    try:
        # Preflight is read-only with respect to all frozen source artifacts.
        save('running', module='mi_localization.convolution_distribution_study', phase='source_preflight')
        subprocess.run([sys.executable,'-m','mi_localization.convolution_distribution_study','--config',args.config,'--preflight'],check=True)
        for module in ("mi_localization.convolution_distribution_study", "mi_localization.convolution_distribution_report"):
            save("running", module=module)
            subprocess.run([sys.executable, "-m", module, "--config", args.config], check=True)
        verified = verified_ready(config)
        if not verified: raise RuntimeError("Convolution-distribution report did not establish complete verification")
        save("complete_verified")
        if args.publish: publish(config); save("complete_published")
    except Exception as error:
        save("publication_failed" if verified else "failed", error=str(error), training_verified=verified); raise
    finally: lock.close()


if __name__ == "__main__": main()
