"use strict";
(() => {
  const data = window.WORLDZERO_DEMO;
  const $ = (id) => document.getElementById(id);
  if (!data) {
    $("chapter-description").textContent =
      "The recording could not load. Reload the page to try again.";
    return;
  }
  const w = data.runs.verify.witness;
  const chapters = [
    {
      time: 0,
      title: "First,<br><em>look around.</em>",
      description:
        "Objects, resources, and no explanation. The agent must learn by acting.",
    },
    {
      time: w.construction,
      title: "An arrangement.<br><em>A possibility.</em>",
      description:
        "The modules form a functional arrangement. A consequence will follow. But one effect is not yet verification.",
    },
    {
      time: w.disruption,
      title: "Take it<br><em>apart.</em>",
      description:
        "After observing a transformation, the agent picks up a module. The suspected mechanism is disrupted.",
    },
    {
      time: w.reconstruction,
      title: "Make it<br><em>happen again.</em>",
      description:
        "The agent reconstructs the arrangement. Watch for the effect to recur, then for the agent to observe it.",
    },
    {
      time: w.benefit,
      title: "Knowledge,<br><em>put to use.</em>",
      description:
        "The agent consumes a resource linked to the recurring effect. The recorded benefit completes the causal chain.",
    },
    {
      time: 160,
      title: "A beginning,<br><em>not an answer.</em>",
      description:
        "The verifier survived and reached Level 4. This is one scripted example, not evidence of general intelligence.",
    },
  ];
  let time = 0,
    playing = false,
    speed = 4,
    policy = "verify",
    comparing = false,
    localView = false;
  let last = performance.now(),
    renderedKey = "",
    activeChapter = -1;
  const names = { verify: "SEARCH + VERIFY", retain: "SEARCH + RETAIN" };
  function frameAt(run, t) {
    let i = 0;
    while (i + 1 < run.frames.length && run.frames[i + 1].time <= t + 1e-7) i++;
    return { frame: run.frames[i], index: i };
  }
  const point = (r, c) => [306 + (c - r) * 24, 52 + (c + r) * 12];
  const diamond = (x, y, rx, ry) =>
    `${x},${y - ry} ${x + rx},${y} ${x},${y + ry} ${x - rx},${y}`;
  function scene(key) {
    const run = data.runs[key],
      { frame: f, index } = frameAt(run, time),
      [ar, ac] = f.position;
    let s = `<svg viewBox="0 0 700 370" xmlns="http://www.w3.org/2000/svg" aria-label="${names[key]}, simulated time ${time.toFixed(1)} seconds" role="img"><title>${names[key]} — recorded world</title><defs><filter id="shadow-${key}"><feGaussianBlur stdDeviation="9"/></filter></defs><ellipse cx="355" cy="291" rx="211" ry="29" fill="#728069" opacity=".12" filter="url(#shadow-${key})"/>`;
    // Exact grid coordinates and recorded resources; height is decorative only.
    for (let sum = 0; sum < 21; sum++)
      for (let r = 0; r < 9; r++) {
        const c = sum - r;
        if (c < 0 || c >= 13) continue;
        const [x, y] = point(r, c),
          fertile = run.fertile[r][c],
          visible =
            !localView || (Math.abs(r - ar) <= 3 && Math.abs(c - ac) <= 3);
        const lift = fertile ? 4 : 0,
          fill = visible ? (fertile ? "#d2dcc5" : "#e5e8dc") : "#e8eadf";
        s += `<g opacity="${visible ? 1 : 0.3}"><path d="M${x - 24},${y} L${x},${y + 12} L${x + 24},${y} V${y + 7} L${x},${y + 19} L${x - 24},${y + 7}Z" fill="${fertile ? "#adbca1" : "#c7d0be"}" stroke="#a9b59f" stroke-width=".5"/><polygon points="${diamond(x, y - lift, 24, 12)}" fill="${fill}" stroke="#a4b39a" stroke-width=".65"/>`;
        if (visible && f.resources[r][c]) {
          const rich = f.resources[r][c] === 2;
          s += rich
            ? `<path d="M${x},${y - 17}l5,7 -5,7 -5,-7Z" fill="#d78443" stroke="#9b4927" stroke-width="1"/><path d="M${x - 7},${y - 9}h-4m18,0h4M${x},${y - 22}v-3" stroke="#d78443" stroke-width="1"/>`
            : `<ellipse cx="${x}" cy="${y - 8}" rx="4" ry="2.5" fill="#768967"/><ellipse cx="${x + 5}" cy="${y - 5}" rx="2" ry="1.5" fill="#9ba985"/>`;
        }
        s += "</g>";
      }
    const trail = run.frames
      .slice(Math.max(0, index - 13), index + 1)
      .map((x) => {
        const p = point(...x.position);
        return `${p[0]},${p[1] - 7}`;
      })
      .join(" ");
    if (!localView)
      s += `<polyline points="${trail}" fill="none" stroke="#b65534" stroke-width="1.2" stroke-dasharray="3 5" opacity=".55"/>`;
    f.modules.forEach((pos, i) => {
      if (!pos) return;
      const [r, c] = pos;
      if (localView && (Math.abs(r - ar) > 3 || Math.abs(c - ac) > 3)) return;
      const [x, y] = point(r, c);
      s += `<g><ellipse cx="${x}" cy="${y + 2}" rx="15" ry="5" fill="#60724f" opacity=".18"/><path d="M${x - 12},${y - 20}l12,6 12,-6v17l-12,6 -12,-6Z" fill="#657b5b" stroke="#43573f" stroke-width=".7"/><path d="M${x},${y - 14}v17l12,-6v-17Z" fill="#8da080"/><polygon points="${diamond(x, y - 20, 12, 6)}" fill="#d5dfc6" stroke="#566a4c" stroke-width=".8"/><text x="${x}" y="${y - 29}" text-anchor="middle" font-size="9" font-family="monospace" fill="#53654a">${String.fromCharCode(65 + i)}</text></g>`;
    });
    const [x, y] = point(ar, ac);
    s += `<g><ellipse cx="${x}" cy="${y}" rx="15" ry="7" fill="#b94820" opacity=".13"/><ellipse cx="${x}" cy="${y - 2}" rx="11" ry="5" fill="none" stroke="#b94820" stroke-width="1.2"/><path d="M${x},${y - 5}v-17" stroke="#8a3821" stroke-width="2"/><circle cx="${x}" cy="${y - 25}" r="7" fill="#bd4b26"/><circle cx="${x - 2}" cy="${y - 27}" r="2" fill="#ffd4a6"/>${f.inventory !== null ? `<path d="M${x + 12},${y - 29}l5,3v6l-5,3 -5,-3v-6Z" fill="#788c65"/>` : ""}</g>`;
    s += `<text x="351" y="340" text-anchor="middle" font-size="9" letter-spacing="2" font-family="monospace" fill="#57674f">${comparing ? names[key] + " / ENERGY " + f.energy.toFixed(1) : "13 COLUMNS × 9 ROWS"}</text></svg>`;
    return s;
  }
  function render() {
    const run = data.runs[policy],
      { frame, index } = frameAt(run, time);
    const chapter = chapters.reduce(
      (best, c, i) => (time + 1e-7 >= c.time ? i : best),
      0,
    );
    if (chapter !== activeChapter) {
      activeChapter = chapter;
      $("chapter-title").innerHTML = chapters[chapter].title;
      $("chapter-description").textContent =
        policy === "retain"
          ? "The chapter markers follow the verifier. This matched retain strategy searches for a useful arrangement, then keeps it instead of deliberately taking it apart."
          : chapters[chapter].description;
      $("chapter-index").textContent = `0${chapter + 1} / 06`;
      document.querySelectorAll("[data-chapter]").forEach((b, i) => {
        b.classList.toggle("active", i === chapter);
        if (i === chapter) b.setAttribute("aria-current", "step");
        else b.removeAttribute("aria-current");
      });
    }
    const renderKey = [
      policy,
      index,
      comparing,
      localView,
      comparing ? frameAt(data.runs.retain, time).index : "",
      comparing ? frameAt(data.runs.verify, time).index : "",
    ].join(":");
    if (renderKey !== renderedKey) {
      $("world").innerHTML = comparing
        ? scene("verify") + scene("retain")
        : scene(policy);
      renderedKey = renderKey;
    }
    $("world").setAttribute(
      "aria-label",
      `${comparing ? "Two matched strategies" : names[policy]}, time ${time.toFixed(1)} seconds. Agent at row ${frame.position[0]}, column ${frame.position[1]}. Energy ${frame.energy.toFixed(1)}.`,
    );
    $("energy").textContent = frame.energy.toFixed(1);
    $("energy-fill").style.width =
      Math.min(100, Math.max(0, (frame.energy / 30) * 100)) + "%";
    $("action").textContent =
      `${frame.action.type}${frame.action.direction ? " / " + frame.action.direction : ""} · FRAME ${index.toString().padStart(3, "0")}`;
    $("time").textContent = "T + " + time.toFixed(2).padStart(6, "0");
    $("timeline").value = time;
    $("timeline").setAttribute(
      "aria-valuetext",
      `${time.toFixed(1)} of 160 simulated seconds`,
    );
    const witness = run.witness;
    document.querySelectorAll("[data-stage]").forEach((el) => {
      const t = witness?.[el.dataset.stage],
        complete = t !== undefined && time + 1e-7 >= t;
      el.classList.toggle("complete", complete);
      el.querySelector(".evidence-symbol").textContent = complete ? "↗" : "○";
      el.querySelector(".evidence-time").textContent = complete
        ? t.toFixed(1) + "s"
        : "—";
    });
    $("evidence-footnote").textContent =
      policy === "retain"
        ? time >= 160
          ? "Episode complete: survived · Level 2. No qualifying reconstruction witness."
          : "This strategy has no qualifying reconstruction witness. The ledger tracks the validated causal chain."
        : time >= 160
          ? "Episode complete: survived · Level 4 under levels-v3. Transfer was not evaluated."
          : time >= w.benefit
            ? "Linked benefit recorded. The agent must still survive to complete the level requirements."
            : time >= w.recurrence_observation
              ? "Reconstruction and recurrence observed. A linked benefit is the next piece of evidence."
              : "Evidence appears as the recorded sequence unfolds.";
    $("play-icon").textContent = playing ? "Ⅱ" : time >= 160 ? "↺" : "▶";
    $("chamber-play").textContent = playing ? "Ⅱ" : time >= 160 ? "↺" : "▶";
    $("chamber-play").setAttribute(
      "aria-label",
      playing
        ? "Pause observation in the world"
        : "Begin observation in the world",
    );
    $("play-label").textContent = playing
      ? "Pause observation"
      : time >= 160
        ? "Watch again"
        : time > 0
          ? "Resume observation"
          : "Begin observation";
    $("play").setAttribute(
      "aria-label",
      playing
        ? "Pause recording"
        : time >= 160
          ? "Replay recording"
          : "Play recording",
    );
  }
  function seek(t) {
    time = Math.min(160, Math.max(0, t));
    render();
  }
  function togglePlay() {
    if (time >= 160) time = 0;
    playing = !playing;
    last = performance.now();
    render();
  }
  $("play").addEventListener("click", togglePlay);
  $("chamber-play").addEventListener("click", togglePlay);
  $("restart").addEventListener("click", () => {
    playing = false;
    seek(0);
  });
  $("timeline").addEventListener("input", (e) => seek(Number(e.target.value)));
  $("speed").addEventListener("click", () => {
    speed = { 1: 4, 4: 8, 8: 1 }[speed];
    $("speed").innerHTML = speed + "× <span>speed</span>";
    $("speed").setAttribute("aria-label", `Playback speed: ${speed} times`);
  });
  $("strategy").addEventListener("change", (e) => {
    policy = e.target.value;
    activeChapter = -1;
    renderedKey = "";
    $("specimen-label").textContent = comparing
      ? "MATCHED SEED / SYNCHRONIZED TIME"
      : "FIG. 01 — " + names[policy];
    render();
  });
  $("compare").addEventListener("click", () => {
    comparing = !comparing;
    $("compare").setAttribute("aria-pressed", comparing);
    $("compare").innerHTML = comparing
      ? "Single observation <span>↗</span>"
      : "Compare the two <span>⇄</span>";
    $("chamber").classList.toggle("comparison", comparing);
    $("specimen-label").textContent = comparing
      ? "MATCHED SEED / SYNCHRONIZED TIME"
      : "FIG. 01 — " + names[policy];
    render();
  });
  $("view-toggle").addEventListener("click", () => {
    localView = !localView;
    $("view-toggle").setAttribute("aria-pressed", localView);
    $("view-label").textContent = localView
      ? "LOCAL VISIBILITY MASK"
      : "OBSERVER VIEW";
    render();
  });
  document.querySelectorAll("[data-chapter]").forEach((b) =>
    b.addEventListener("click", () => {
      playing = false;
      seek(chapters[Number(b.dataset.chapter)].time);
    }),
  );
  const dialog = $("information");
  function showDialog(kind) {
    playing = false;
    render();
    $("dialog-eyebrow").textContent =
      kind === "rule"
        ? "FOR THE OBSERVER / HIDDEN FROM THE AGENT"
        : "RESEARCH NOTES / SPECIMEN 001";
    $("dialog-title").textContent =
      kind === "rule"
        ? "A useful arrangement changes the world."
        : "A world with no instructions for its hidden rule.";
    $("dialog-body").innerHTML =
      kind === "rule"
        ? `<div class="rule-diagram">MODULE B + MODULE C, IN THE RIGHT RELATION<br>↓<br>NEARBY RAW RESOURCE → RICH RESOURCE</div><p>This recorded world contains catalysis. An eligible module arrangement enables nearby resources to transform. The agent is never told the relevant pair or relation.</p><p>The verifier discovers a useful arrangement, disrupts it, rebuilds it, observes the effect recur, and consumes a linked output. The study validates that chain against the trace.</p><p>Letters, resource colors, and the full map are observer aids. The local visibility mask shows the agent’s observation radius; it does not reproduce the agent’s exact input format.</p><p><a href="${data.runs[policy].trace_url}">Inspect the original trace ↗</a></p>`
        : `<p>WorldZero tests whether an agent can discover and control an unfamiliar causal mechanism before its finite lifetime ends.</p><p>You are watching recorded scripted policies, with no live model calls. Both strategies use seed 408557419 under the same pressure budgets. Playback shows the most recent recorded action state at each simulated time; it does not invent intermediate physics.</p><p>The verifier’s chapters come from the validated causal witness. The evidence ledger tracks that chain, rather than assigning a score before the episode finishes.</p><p>The full study covers 192 episodes, including matched null worlds. This selected example illustrates a successful verifier; it is not representative of every run.</p><p><a href="https://github.com/skishore23/worldzero/blob/main/evidence/verification-study/README.md">Read the study ↗</a> · <a href="https://github.com/skishore23/worldzero/blob/main/evidence/verification-rescore/README.md">Read the scoring correction ↗</a></p>`;
    dialog.showModal();
  }
  $("reveal").addEventListener("click", () => showDialog("rule"));
  document
    .querySelector("[data-dialog]")
    .addEventListener("click", () => showDialog("about"));
  document
    .querySelector(".dialog-close")
    .addEventListener("click", () => dialog.close());
  dialog.addEventListener("click", (e) => {
    if (e.target === dialog) {
      const r = dialog.getBoundingClientRect();
      if (
        e.clientX < r.left ||
        e.clientX > r.right ||
        e.clientY < r.top ||
        e.clientY > r.bottom
      )
        dialog.close();
    }
  });
  document.addEventListener("keydown", (e) => {
    if (
      e.code === "Space" &&
      !dialog.open &&
      !["INPUT", "BUTTON", "SELECT", "A", "TEXTAREA"].includes(e.target.tagName)
    ) {
      e.preventDefault();
      togglePlay();
    }
  });
  document.addEventListener("visibilitychange", () => {
    last = performance.now();
  });
  function tick(now) {
    if (playing && !document.hidden) {
      time = Math.min(160, time + Math.min((now - last) / 1000, 0.1) * speed);
      if (time >= 160) playing = false;
      render();
    }
    last = now;
    requestAnimationFrame(tick);
  }
  render();
  requestAnimationFrame(tick);
})();
