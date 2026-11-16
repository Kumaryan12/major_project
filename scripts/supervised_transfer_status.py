"""Read-only matched-transfer progress; file counters are not verified metrics."""
import argparse
import json
from pathlib import Path
import time

import yaml

from supervised_source_status import command_for_pid


def status(config):
    root=Path(config['output_dir'])/'complete';path=root/'progress.json'
    saved=json.loads(path.read_text()) if path.exists() else {};command=command_for_pid(saved.get('pid'))
    alive=bool(command and any(n in command for n in ['mi_localization.supervised_transfer','scripts/run_supervised_transfer.py']))
    expected=len(config['arms'])*len(config['training_seeds'])*config['outer_splits']*(len(config['learning_rates'])*config['inner_splits']+1)
    files=list((root/'review/fits').glob('*/audit.json'));outer=sum(p.parent.name.endswith('_outer') for p in files)
    lines=[f'Matched supervised-source transfer: {root}',f"State: {saved.get('state','not started')} | local command matches: {alive}",
           f"Updated UTC: {saved.get('updated_utc','unavailable')}",f'Locally saved fit audits: {len(files)}/{expected}',
           f"Outer models saved: {outer}/{len(config['arms'])*len(config['training_seeds'])*config['outer_splits']}"]
    if 'arm' in saved:
        lines.append(f"Current: seed {saved['training_seed']}, fold {saved['outer_fold']}, {saved['arm']}, {saved.get('fit_id','outer')}, epoch {saved.get('epoch','—')}")
    lines.extend([f"Final report present: {(root/'REPORT.md').exists()}",
                  'Three target arms/all seeds; partial counters/inner scores are not a final patient macro F1.',
                  'Study stops after independent restoration/report/publication. Patient identity remains unresolved.'])
    return '\n'.join(lines)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/supervised_transfer_v1.yaml');p.add_argument('--watch',action='store_true')
    args=p.parse_args();c=yaml.safe_load(Path(args.config).read_text())
    try:
        while True:
            if args.watch: print('\033[2J\033[H',end='')
            print(status(c),flush=True)
            if not args.watch: break
            time.sleep(10)
    except KeyboardInterrupt: pass
