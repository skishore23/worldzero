"""The public visual projection must remain tied to the original experiment."""
import json
from pathlib import Path
from scripts.build_browser_demo import build
from worldzero.protocol import read_trace

ROOT = Path(__file__).resolve().parents[1]


def test_demo_projection_matches_authenticated_trace_and_published_asset():
    data = build()
    payload = (ROOT / 'demo/data.js').read_text()
    assert json.loads(payload.removeprefix('window.WORLDZERO_DEMO = ').removesuffix(';\n')) == data
    for policy, run in data['runs'].items():
        trace = read_trace(ROOT / f'evidence/verification-study/traces/{policy}-worldzero-catalysis-active/408557419.json.gz')
        assert len(run['frames']) == len(trace['decisions']) + 1
        assert run['frames'][0]['time'] == 0
        for frame, decision in zip(run['frames'][1:], trace['decisions']):
            original = decision['post_display']
            assert frame['resources'] == original['resources']
            assert frame['modules'] == original['modules']
            assert frame['position'] == original['agent']['position']
            assert frame['energy'] == original['agent']['energy']
        assert run['frames'][-1]['time'] == run['result']['world_time']
    witness = data['runs']['verify']['witness']
    assert witness['construction'] < witness['disruption'] < witness['reconstruction']
    assert witness['reconstruction'] < witness['recurrence_observation'] < witness['benefit']
    assert data['runs']['retain']['witness'] is None
