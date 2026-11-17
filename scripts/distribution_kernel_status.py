"""Read-only PID/command-checked status; Ctrl-C stops only the monitor."""
import argparse
import json
from pathlib import Path
import time
import yaml
from supervised_source_status import command_for_pid


def read(p):return json.loads(p.read_text()) if p.is_file() else {}


def status(config):
    base=Path(config['output_dir']);root=base/'complete';runner=read(base/'runner.json');saved=read(root/'progress.json');s=read(root/'summary.json')
    command=command_for_pid(saved.get('pid'));rc=command_for_pid(runner.get('pid'))
    alive=bool(command and any(x in command for x in ('mi_localization.distribution_kernel_study','mi_localization.distribution_kernel_report')))
    groups=config['outer_splits']*len(config['projection_seeds']);r=len(config['representations']);linear=groups*r*(config['inner_splits']*len(config['ridge_alphas'])+1);nonlinear=groups*r*(config['inner_splits']*len(config['ridge_alphas'])*len(config['nonlinear_weights'])+1)
    fits=list((root/'review/fits').glob('*.json'));eq=list((root/'review/equivalence').glob('*.json'));ctx=list((root/'review/contexts').glob('*.json'))
    lines=[f"Kernel readouts: {runner.get('state','not started')} | matching local runner: {bool(rc and 'run_distribution_kernel.py' in rc)}",
           f"Phase: {runner.get('phase','unavailable')} | worker state: {saved.get('state','not started')} | matching local worker: {alive}",
           f"Saved classifiers: {len(fits)}/{linear+nonlinear} ({sum('__linear_' in p.name for p in fits)}/{linear} linear, {sum('__nonlinear_' in p.name for p in fits)}/{nonlinear} nonlinear)",
           f"Saved outer models: {sum(p.name.startswith('outer_') for p in fits)}/{groups*r*2}",
           f"Kernel contexts: {len(ctx)}/{groups*(config['inner_splits']+1)*r} | source equivalence receipts: {len(eq)}/{linear}",
           f"Updated UTC: {saved.get('updated_utc','unavailable')}"]
    if 'outer_fold' in saved:lines.append(f"Current fold {saved['outer_fold']}, seed {saved.get('projection_seed')}, inner {saved.get('inner_fold')}, {saved.get('representation')}, {saved.get('family')}")
    lines.extend([f"Validated summary recorded: {s.get('complete') is True}",'Saved counts, linear gate and engineering smoke are not completed nonlinear performance. No new waveform extraction.'])
    return '\n'.join(lines)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/distribution_kernel_v1.yaml');p.add_argument('--watch',action='store_true');args=p.parse_args();config=yaml.safe_load(Path(args.config).read_text())
    try:
        while True:
            if args.watch:print('\033[2J\033[H',end='')
            print(status(config),flush=True)
            if not args.watch:break
            time.sleep(10)
    except KeyboardInterrupt:pass
