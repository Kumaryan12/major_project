"""Portable, read-only source pilot monitor. No numerical imports or new work."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def command_for_pid(pid):
    try:
        if type(pid) is not int or pid<=0: return ''
        if os.name=='posix':
            result=subprocess.run(['ps','-p',str(pid),'-o','command='],capture_output=True,text=True,timeout=5)
        else:
            result=subprocess.run(['powershell','-NoProfile','-Command',
                f'(Get-CimInstance Win32_Process -Filter "ProcessId = {pid}").CommandLine'],capture_output=True,text=True,timeout=10)
        return result.stdout.strip() if result.returncode==0 else ''
    except (OSError,subprocess.TimeoutExpired): return ''


def snapshot(root):
    path=root/'progress.json'
    progress=json.loads(path.read_text()) if path.exists() else {}
    command=command_for_pid(progress.get('pid'))
    alive=bool(command and any(s in command for s in ['mi_localization.supervised_source_bank','mi_localization.supervised_source_train']))
    fits=list((root/'runs').glob('*/summary.json'))
    rows=[f'Supervised source-only pilot: {root}',f"Saved state: {progress.get('state','not started')}",
          f"Local worker command matches: {alive}",f"Last update (UTC): {progress.get('updated_utc','unavailable')}"]
    if 'completed' in progress:
        rows.append(f"Records: {progress['completed']}/{progress['total']} ({100*progress['completed']/progress['total']:.1f}%)")
    if 'objective' in progress:
        rows.append(f"Objective {progress['objective']}, seed {progress['seed']}, epoch {progress['epoch']}; validation macro AP {progress['validation_macro_ap']:.5f}")
    rows.extend([f'Completed source fits: {len(fits)}',f"Final source report present: {(root/'REPORT.md').exists()}",
                 'Source annotation metrics are not PTB six-class patient macro F1. No downstream transfer is queued.'])
    return '\n'.join(rows)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',default='results/supervised_source_pilot_v1');p.add_argument('--watch',action='store_true')
    args=p.parse_args()
    try:
        while True:
            if args.watch: print('\033[2J\033[H',end='')
            print(snapshot(Path(args.output_dir)),flush=True)
            if not args.watch: break
            time.sleep(10)
    except KeyboardInterrupt: pass
