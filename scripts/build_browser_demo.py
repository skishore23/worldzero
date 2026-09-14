"""Export a small, authenticated display projection of two published study traces."""
from pathlib import Path
import json
from worldzero.protocol import read_trace
from worldzero.causal_evidence import benchmark_evidence, public_trace_events
from worldzero.util import digest

ROOT = Path(__file__).resolve().parents[1]

def build():
    study = ROOT / 'evidence/verification-study'
    rows = json.loads((study / 'results.json').read_text())['rows']
    runs = {}
    for policy in ('verify', 'retain'):
        row = next(r for r in rows if r['policy'] == policy and r['family_id'] == 'worldzero:catalysis'
                   and r['seed'] == 408557419 and r['arm'] == 'active')
        trace = read_trace(study / row['trace']['path'])
        if digest(trace) != row['trace']['sha256']:
            raise ValueError('Published trace changed')
        states = [trace['states'][0]] + [d['post_display'] for d in trace['decisions']]
        frames = []
        for s in states:
            a = s['agent']
            frames.append({'time': s['time'], 'position': a['position'], 'energy': a['energy'],
                           'inventory': a['inventory'], 'modules': s['modules'], 'resources': s['resources'],
                           'conversions': s['conversions'], 'motif': s['motif'],
                           'action': a['last_result'].get('action', {'type': 'OBSERVE'}),
                           'rich_consumed': a['rich_consumed']})
        witness = benchmark_evidence(trace)['stage_evidence']['causal_witness']
        events = public_trace_events(trace)
        runs[policy] = {'frames': frames, 'fertile': states[0]['fertile'], 'result': row['episode'],
                        'trace_sha256': row['trace']['sha256'],
                        'trace_url': 'https://github.com/skishore23/worldzero/blob/main/evidence/verification-study/' + row['trace']['path'],
                        'witness': {k: (events[v]['time'] if 'time' in events[v] else events[v]['observation']['time']) for k,v in witness.items() if v is not None} if witness else None}
    return {'schema': 'worldzero-browser-demo-v1', 'seed': 408557419, 'family': 'worldzero:catalysis',
            'kind': 'recorded_scripted_experiment', 'runs': runs}

if __name__ == '__main__':
    output = ROOT / 'demo/data.js'
    output.write_text('window.WORLDZERO_DEMO = ' + json.dumps(build(), separators=(',', ':')) + ';\n')
    print(output.relative_to(ROOT))
