"""Summarize a local call from content-free diagnostic logs."""
import argparse
import json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--call-id');args=p.parse_args()
root=Path(__file__).resolve().parents[1]/'.runtime'
records=[]
for path in root.glob('call-events.jsonl*'):
    for line in path.read_text().splitlines():
        try: records.append(json.loads(line))
        except ValueError: pass
records.sort(key=lambda r:r['time'])
call=args.call_id or next((r.get('call_id') for r in reversed(records) if r.get('call_id')),None)
if not call:
    print('No call diagnostics yet. Refresh the app and try a call.');raise SystemExit()
matching=[r for r in records if r.get('call_id')==call]
print(f'Call {call}')
for r in matching:
    if r['event'] in ('server_start','server_end','client_end','socket_close','audio_state','microphone_mute','microphone_unmute'):
        print(json.dumps(r,indent=2))
samples=[r for r in matching if r['event'] in ('client_sample','server_sample')]
if samples:
    print('Last timing/device samples:')
    for r in samples[-4:]: print(json.dumps(r,indent=2))
