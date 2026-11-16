"""Read-only native-pilot and queued expanded-audit status."""
import argparse
import json
from pathlib import Path
import time

from supervised_source_status import snapshot,command_for_pid


def status():
    lines=[snapshot(Path('results/supervised_source_pilot_v1')),'','Next: expanded source recording-overlap audit']
    root=Path('results/supervised_source_overlap_v1')
    for name in ['queue_progress.json','progress.json']:
        path=root/name
        if not path.exists(): continue
        s=json.loads(path.read_text());command=command_for_pid(s.get('pid'))
        running=bool(command and any(n in command for n in ['queue_supervised_source_overlap.py','mi_localization.supervised_source_overlap']))
        lines.append(f"{name}: {s['state']} | local command matches: {running} | updated: {s['updated_utc']}")
        if 'completed' in s: lines.append(f"  {s['completed']}/{s['total']}")
    lines.append('This queue ends after the expanded audit. No target classification/transfer is queued.')
    return '\n'.join(lines)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--watch',action='store_true');args=p.parse_args()
    try:
        while True:
            if args.watch: print('\033[2J\033[H',end='')
            print(status(),flush=True)
            if not args.watch: break
            time.sleep(10)
    except KeyboardInterrupt: pass
