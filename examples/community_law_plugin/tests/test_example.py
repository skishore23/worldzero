"""Offline wheel/install/discovery/run/replay smoke test for the example plugin."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


REPOSITORY = Path(__file__).resolve().parents[3]
EXAMPLE = Path(__file__).resolve().parents[1]


def run(*command: str, cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )


def test_example_wheel_installs_validates_runs_and_replays_offline(tmp_path: Path) -> None:
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    run(
        sys.executable,
        "-m",
        "build",
        "--wheel",
        "--no-isolation",
        "--outdir",
        str(wheels),
        cwd=REPOSITORY,
    )
    run(
        sys.executable,
        "-m",
        "build",
        "--wheel",
        "--no-isolation",
        "--outdir",
        str(wheels),
        cwd=EXAMPLE,
    )
    worldzero_wheel = next(wheels.glob("worldzero_research-*.whl"))
    example_wheel = next(wheels.glob("worldzero_example_law-*.whl"))

    environment = tmp_path / "venv"
    run(sys.executable, "-m", "venv", "--system-site-packages", str(environment), cwd=tmp_path)
    python = environment / "bin" / "python"
    run(
        str(python),
        "-m",
        "pip",
        "install",
        "--no-index",
        "--no-deps",
        str(worldzero_wheel),
        str(example_wheel),
        cwd=tmp_path,
    )
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    program = """
import json
from worldzero.experiment import run_episode, verify_plugin_replay
from worldzero.kernel import Config, World
from worldzero.laws import FamilyTestKit, installed_registry

class Wait:
    name = 'example-smoke'
    def decide(self, observation):
        return {'action': {'type': 'WAIT', 'duration': 0.1}, 'memory': ''}

registry = installed_registry()
assert 'example_org:preserver' in registry.list_family_ids()
registered = registry.resolve('example_org:preserver')
assert registered.official is False and registered.experimental is True
report = FamilyTestKit(registry).validate('example_org:preserver', seeds=range(2))
assert report['passed'], report
world = World(19, Config(max_decisions=1), family=registered, record=True)
result, trace = run_episode(world, Wait(), capture=True)
assert result['status'] == 'censored'
assert trace['family_identity']['experimental'] is True
replay = verify_plugin_replay(trace, registry=registry)
assert replay['verified'] is True
print(json.dumps({'report': report, 'replay': replay}, sort_keys=True))
"""
    completed = run(str(python), "-c", program, cwd=tmp_path, env=env)
    payload = json.loads(completed.stdout)
    assert payload["report"]["family_id"] == "example_org:preserver"
    assert payload["replay"]["verified"] is True

    listed = run(str(python), "-m", "worldzero", "laws", "list", cwd=tmp_path, env=env)
    assert "example_org:preserver" in json.loads(listed.stdout)["family_ids"]
    inspected = run(
        str(python),
        "-m",
        "worldzero",
        "laws",
        "inspect",
        "example_org:preserver",
        "--experimental-family",
        cwd=tmp_path,
        env=env,
    )
    assert json.loads(inspected.stdout)["experimental"] is True
    assert "trusted in-process Python" in inspected.stderr
