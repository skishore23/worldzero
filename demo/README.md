# WorldZero field observatory

A static, interactive replay of two actual scripted trajectories from the
published verification study. No build tools, external fonts, API keys, network
requests, or live inference are needed for playback.

From the repository root:

```bash
python -m http.server 8771 --bind 127.0.0.1 --directory demo
```

Open <http://127.0.0.1:8771>. The page also works by opening `index.html` directly.
Publish the contents of this directory to any static host to share it publicly.

Controls: play/pause, restart, scrub, speed, six chapter jumps, strategy selector,
synchronized comparison, local visibility mask, and hidden-rule reveal. Space
plays/pauses when focus is outside an interactive control. Playback does not
autostart. The native dialog supports Escape and restores focus.

## Evidence and display

`data.js` contains a compact display projection of the verify and retain active
catalysis runs at seed 408557419. The exporter authenticates each source trace
against its canonical digest in the published study. It copies recorded positions,
resources, modules, energy, and actions, and derives chapter timestamps from the
validated causal witness. Regenerate it from the repository root:

```bash
python -m scripts.build_browser_demo
```

The isometric height, shading, object letters, colors, and trail are observer
presentation aids. Playback holds the most recent action frame; it does not
interpolate or simulate physics. The local visibility mask uses the recorded
radius of three cells; it is not the agent's exact input representation. Hidden
rule details are explained in the observer-only reveal.

Chapter markers always follow the verifier, including during comparison.
The evidence ledger tracks the qualifying reconstruction witness; a blank
retain ledger does not imply that no arrangement or effect ever occurred.
Scores are only reported at episode completion. These are selected scripted
examples, not model-discovery results or a representative sample of the study.

The original experimental files remain unchanged.
