const SENSOR_LABELS = ["Thumb", "Index", "Middle", "Ring", "Pinky", "Palm", "Wrist", "Ulnar"];
const SENSOR_COLORS = ["#ef4444", "#f97316", "#eab308", "#22c55e", "#06b6d4", "#3b82f6", "#8b5cf6", "#ec4899"];

const DEFAULT_VARIANT = "v10";
const VARIANT_TARGETS = { v10: 105, v5: 55 };

const state = {
  username: localStorage.getItem("glovesense.username") || "",
  variant: localStorage.getItem("glovesense.variant") || DEFAULT_VARIANT,
  variants: {},
  ws: null,
  pauseRequested: false,
  cookieCount: 0,
  mapping: {},
  letters: [],
  charts: [],
  recordings: {},   // { [gestureId]: count }
  activeGesture: null,
  models: [],
  selectedModel: localStorage.getItem("glovesense.model") || "rf",
  featureSet: localStorage.getItem("glovesense.featureSet") || "basic",
};

const el = (id) => document.getElementById(id);

function setStatus(node, text, kind = "") {
  if (!node) return;
  node.textContent = text;
  node.classList.remove("error", "success", "warn");
  if (kind) node.classList.add(kind);
}

function setPill(node, text, kind = "") {
  node.textContent = text;
  node.classList.remove("ok", "err", "active");
  if (kind) node.classList.add(kind);
}

function setPhase(phase, label = null) {
  const ring = document.querySelector(".phase-ring");
  const chip = el("phaseChip");
  ring.dataset.phase = phase;
  chip.dataset.phase = phase;
  chip.textContent = label ?? phase.charAt(0).toUpperCase() + phase.slice(1);
}

function setMode(label, kind = "") {
  const pill = el("modePill");
  pill.dataset.activity = label;
  setPill(pill, label, kind);
  updateModePill();
}

function setProgress(count, target) {
  const pct = target > 0 ? Math.min(100, Math.round((count / target) * 100)) : 0;
  el("progressFill").style.width = `${pct}%`;
  el("progressText").textContent = target > 0 ? `${count} / ${target} samples` : "0 / 0 samples";
}

function toast(message, kind = "") {
  const node = document.createElement("div");
  node.className = `toast${kind ? " " + kind : ""}`;
  node.textContent = message;
  el("toasts").appendChild(node);
  setTimeout(() => {
    node.classList.add("fade");
    setTimeout(() => node.remove(), 300);
  }, 3500);
}

async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try { const j = await res.json(); detail = j.detail || detail; } catch {}
    throw new Error(detail);
  }
  return res.json();
}

function initCharts() {
  state.charts = SENSOR_LABELS.map((label, i) => {
    const ctx = el(`chart${i + 1}`).getContext("2d");
    return new Chart(ctx, {
      type: "line",
      data: {
        labels: [],
        datasets: [{
          label,
          data: [],
          borderColor: SENSOR_COLORS[i],
          backgroundColor: SENSOR_COLORS[i] + "22",
          borderWidth: 1.8,
          pointRadius: 0,
          tension: 0.25,
          fill: true,
        }],
      },
      options: {
        animation: false,
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { display: false },
          y: {
            ticks: { font: { size: 9 }, color: "#94a3b8" },
            grid: { color: "rgba(15,23,42,0.05)" },
          },
        },
      },
    });
  });
}

function resetCharts() {
  state.charts.forEach((c) => {
    c.data.labels = [];
    c.data.datasets[0].data = [];
    c.update("none");
  });
}

function pushSample(sensorIndex, value) {
  const c = state.charts[sensorIndex];
  c.data.labels.push(c.data.labels.length);
  c.data.datasets[0].data.push(value);
  if (c.data.labels.length > 250) {
    c.data.labels.shift();
    c.data.datasets[0].data.shift();
  }
  c.update("none");
}

function setGestureImage(n) {
  const img = el("gestureImg");
  img.src = `/gestures/${n}.jpg`;
  img.onerror = () => { img.style.visibility = "hidden"; };
  img.onload = () => { img.style.visibility = ""; };
  const letter = state.mapping[String(n)] ?? (n === 0 ? "RS" : "?");
  el("gestureLetter").textContent = letter;
}

async function refreshUsers() {
  try {
    const { users } = await api("/api/users");
    const list = el("userList");
    list.innerHTML = "";
    if (users.length === 0) {
      const li = document.createElement("li");
      li.className = "empty";
      li.textContent = "No profiles yet";
      list.appendChild(li);
    } else {
      users.forEach((u) => {
        const li = document.createElement("li");
        li.textContent = u;
        if (u === state.username) li.classList.add("active");
        li.addEventListener("click", async () => {
          state.username = u;
          localStorage.setItem("glovesense.username", u);
          el("username").value = u;
          updateUserPill();
          await refreshUsers();
          await refreshGestureGrid();
        });
        list.appendChild(li);
      });
    }
  } catch (e) {
    setStatus(el("setupStatus"), `Failed to load profiles: ${e.message}`, "error");
  }
}

function updateUserPill() {
  setPill(el("userPill"), state.username ? `👤 ${state.username}` : "No user", state.username ? "active" : "");
}

async function loadGestureMapping() {
  const { mapping, variants, default_variant } = await api("/api/gestures");
  state.mapping = mapping;
  state.variants = variants || {};
  if (!state.variants[state.variant]) {
    state.variant = default_variant || DEFAULT_VARIANT;
  }
  state.letters = Object.entries(mapping)
    .sort(([a], [b]) => Number(a) - Number(b))
    .map(([k, v]) => ({ id: Number(k), letter: v }));
  const populate = (selectId) => {
    const sel = el(selectId);
    sel.innerHTML = "";
    state.letters.forEach(({ id, letter }) => {
      const opt = document.createElement("option");
      opt.value = id;
      opt.textContent = `${letter} · #${id}`;
      sel.appendChild(opt);
    });
  };
  populate("selRepeat");
  populate("selTest");
  renderVariantToggle();
  renderGestureGrid();
}

async function loadModelCatalog() {
  try {
    const { models, default_model } = await api("/api/models");
    state.models = models || [];
    if (!state.models.find((m) => m.key === state.selectedModel)) {
      state.selectedModel = default_model || "rf";
    }
    const sel = el("selModel");
    if (sel) {
      sel.innerHTML = "";
      state.models.forEach(({ key, display }) => {
        const opt = document.createElement("option");
        opt.value = key;
        opt.textContent = display;
        sel.appendChild(opt);
      });
      sel.value = state.selectedModel;
    }
  } catch (e) {
    /* ignore */
  }
}

function renderVariantToggle() {
  const toggle = el("variantToggle");
  if (!toggle) return;
  toggle.querySelectorAll("button").forEach((btn) => {
    const v = btn.dataset.variant;
    const meta = state.variants[v];
    if (meta) btn.textContent = `${meta.label} · ${meta.train_samples} samples`;
    btn.classList.toggle("active", v === state.variant);
    btn.setAttribute("aria-checked", v === state.variant ? "true" : "false");
  });
  updateModePill();
}

function updateModePill() {
  const pill = el("modePill");
  if (!pill) return;
  const baseMode = state.ws ? (pill.dataset.activity || "Busy") : "Idle";
  const variantLabel = state.variants[state.variant]?.label ?? state.variant;
  pill.textContent = `${baseMode} · ${variantLabel}`;
}

function lockVariantToggle(locked) {
  el("variantToggle").querySelectorAll("button").forEach((b) => { b.disabled = locked; });
  const hint = el("variantHint");
  if (hint) {
    hint.textContent = locked
      ? "Profile locked during recording — finish or abort to change it."
      : "Switch any time between recordings.";
    hint.classList.toggle("warn", locked);
  }
}

function renderGestureGrid() {
  const ul = el("gestureGrid");
  ul.innerHTML = "";
  state.letters.forEach(({ id, letter }) => {
    const li = document.createElement("li");
    const count = state.recordings[id] || 0;
    li.innerHTML = `<span>${letter}</span><span class="reps">${count}×</span>`;
    li.title = `Gesture ${id} (${letter}) — recorded ${count} time${count === 1 ? "" : "s"}`;
    li.dataset.id = id;
    if (count > 0) li.classList.add("has");
    if (state.activeGesture === id) li.classList.add("current");
    li.addEventListener("click", () => { el("selRepeat").value = id; });
    ul.appendChild(li);
  });
}

async function refreshGestureGrid() {
  state.recordings = {};
  if (!state.username) {
    renderGestureGrid();
    return;
  }
  try {
    const { gestures } = await api(`/api/users/${encodeURIComponent(state.username)}/gestures`);
    gestures.forEach((g) => { state.recordings[g.n] = g.recordings || 0; });
  } catch (e) { /* ignore */ }
  renderGestureGrid();
}

function ensureUser() {
  if (!state.username) {
    setStatus(el("setupStatus"), "Create or pick a profile first.", "error");
    toast("Pick a profile first", "warn");
    return false;
  }
  return true;
}

function clearReaction() {
  const r = el("reaction");
  r.hidden = true;
  r.innerHTML = "";
  r.className = "reaction";
}

function streamCollection(gestureNumber, mode) {
  if (!ensureUser()) return;
  if (state.ws) {
    toast("A collection is already running.", "warn");
    return;
  }
  resetCharts();
  setGestureImage(0);
  setPhase("rest", "Starting…");
  setMode(mode === "train" ? "Training" : "Testing", "active");
  state.pauseRequested = false;
  state.activeGesture = gestureNumber;
  renderGestureGrid();
  const initialTarget = mode === "train" ? (VARIANT_TARGETS[state.variant] ?? 105) : 15;
  setProgress(0, initialTarget);
  lockVariantToggle(true);

  const wsProto = location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(`${wsProto}//${location.host}/ws/collect`);
  state.ws = ws;

  ws.addEventListener("open", () => {
    ws.send(JSON.stringify({
      username: state.username,
      gesture_number: Number(gestureNumber),
      mode,
      variant: state.variant,
    }));
    setStatus(el("collectStatus"), `Collecting ${mode} data for gesture ${gestureNumber}…`);
  });

  ws.addEventListener("message", (msg) => {
    let ev;
    try { ev = JSON.parse(msg.data); } catch { return; }

    if (ev.event === "started") {
      setProgress(0, ev.target_samples ?? 0);
      setStatus(el("collectStatus"), `Recording gesture ${ev.gesture_letter}`);
    } else if (ev.event === "sample") {
      pushSample(ev.sensor - 1, ev.value);
      setProgress(ev.count, ev.target);
      setStatus(el("collectStatus"), `Sample ${ev.count}/${ev.target} (sensor ${ev.sensor})`);
    } else if (ev.event === "phase") {
      setGestureImage(ev.image);
      setPhase(ev.phase, ev.phase === "rest" ? "Rest" : "Hold gesture");
    } else if (ev.event === "paused") {
      setPhase("paused", "Paused");
    } else if (ev.event === "resumed") {
      setPhase("beep", "Hold gesture");
    } else if (ev.event === "done") {
      setPhase("done", "Saved ✓");
      setStatus(el("collectStatus"), "Collection complete.", "success");
      toast(`Saved ${mode} sample for ${state.mapping[gestureNumber] ?? gestureNumber}`, "success");
      if (mode === "train") {
        state.recordings[gestureNumber] = (state.recordings[gestureNumber] || 0) + 1;
        renderGestureGrid();
        refreshGestureGrid();
      }
    } else if (ev.event === "aborted") {
      setPhase("error", "Aborted");
      setStatus(el("collectStatus"), "Collection aborted.", "warn");
      toast("Collection aborted", "warn");
    } else if (ev.event === "error") {
      setPhase("error", "Error");
      setStatus(el("collectStatus"), `Error: ${ev.detail}`, "error");
      toast(`Error: ${ev.detail}`, "error");
    }
  });

  ws.addEventListener("close", () => {
    state.ws = null;
    setMode("Idle");
    state.activeGesture = null;
    renderGestureGrid();
    el("btnPause").textContent = "⏸ Pause";
    lockVariantToggle(false);
  });

  ws.addEventListener("error", () => {
    setStatus(el("collectStatus"), "WebSocket error.", "error");
    toast("WebSocket error", "error");
  });
}

function sendWsAction(action) {
  if (state.ws && state.ws.readyState === WebSocket.OPEN) {
    state.ws.send(JSON.stringify({ action }));
  }
}

function abortCollection() {
  if (state.ws) {
    sendWsAction("abort");
    setStatus(el("collectStatus"), "Abort requested…");
  }
}

function togglePause() {
  if (!state.ws) return;
  state.pauseRequested = !state.pauseRequested;
  sendWsAction(state.pauseRequested ? "pause" : "resume");
  el("btnPause").textContent = state.pauseRequested ? "▶ Resume" : "⏸ Pause";
}

function renderTrainResults(result) {
  const rows = (result.results || [])
    .sort((a, b) => (b.train_accuracy ?? -1) - (a.train_accuracy ?? -1))
    .map((r) => {
      if (r.error) {
        return `<tr><td>${r.display}</td><td colspan="2" class="miss">${r.error}</td></tr>`;
      }
      const pct = (r.train_accuracy * 100);
      const cls = pct >= 90 ? "" : pct >= 65 ? "mid" : "lo";
      return `
        <tr>
          <td>${r.display}</td>
          <td class="lb-bar-cell">
            <div class="lb-bar"><div class="lb-bar-fill ${cls}" style="width:${pct.toFixed(1)}%"></div>
              <div class="lb-bar-label">${pct.toFixed(1)}%</div>
            </div>
          </td>
          <td style="font-size:0.7rem;color:var(--text-mute);font-family:monospace;">${r.path.split(/[\\/]/).pop()}</td>
        </tr>`;
    })
    .join("");

  el("trainResults").innerHTML = `
    <div class="metric-row">
      <div class="metric primary">
        <div class="k">Samples</div>
        <div class="v">${result.samples}</div>
      </div>
      <div class="metric">
        <div class="k">Features</div>
        <div class="v">${result.feature_dim}-D</div>
      </div>
      <div class="metric">
        <div class="k">Classes</div>
        <div class="v">${result.classes_trained?.length ?? 0}</div>
      </div>
      <div class="metric accent">
        <div class="k">Models</div>
        <div class="v">${result.results.length}</div>
      </div>
    </div>
    <table class="lb-table">
      <thead><tr><th>Model</th><th>Training accuracy</th><th>File</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
    <p class="muted" style="margin-top:10px;font-size:0.78rem;">
      Training accuracy is on the same data the model saw — it's an upper bound, not a generalization estimate. Use the Model Lab to see honest cross-validation and leave-one-user-out scores.
    </p>
  `;
}

function renderPrediction(result) {
  clearReaction();
  const reaction = el("reaction");
  reaction.hidden = false;
  reaction.classList.add(result.correct ? "correct" : "incorrect");
  const img = result.correct ? "/gestures/smile.jpg" : "/gestures/sad.jpg";
  const headline = result.correct ? "🎯 Spot on!" : "😅 Not quite…";
  const probsHtml = (result.top_probabilities || [])
    .slice(0, 5)
    .map((p) => `<li><span>${p.letter}</span><span>${(p.p * 100).toFixed(1)}%</span></li>`)
    .join("");
  reaction.innerHTML = `
    <img src="${img}" alt="${result.correct ? "smile" : "sad"}" />
    <h3>${headline}</h3>
    <div class="compare">
      <div>
        <div class="k">Expected</div>
        <div class="v">${result.expected}</div>
      </div>
      <div>
        <div class="k">Predicted</div>
        <div class="v">${result.predicted}</div>
      </div>
      <div>
        <div class="k">Model</div>
        <div class="v" style="font-size:0.95rem;">${result.model_display}</div>
      </div>
    </div>
    ${probsHtml ? `<ul style="list-style:none;padding:0;margin:6px 0 0;display:flex;gap:10px;font-size:0.75rem;color:var(--text-soft);">${probsHtml}</ul>` : ""}
  `;
  toast(result.correct ? `${result.model_display}: ${result.predicted} ✓` : `${result.model_display}: ${result.predicted} ✗`,
        result.correct ? "success" : "warn");
}

function wireTabs() {
  document.querySelectorAll(".tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((b) => b.classList.toggle("active", b === btn));
      const target = btn.dataset.tab;
      document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("active", p.id === `${target}Panel`));
    });
  });
}

function wireSetup() {
  el("username").value = state.username;
  updateUserPill();

  el("btnCreateFolder").addEventListener("click", async () => {
    const name = el("username").value.trim();
    if (!name) {
      setStatus(el("setupStatus"), "Enter a profile name.", "error");
      return;
    }
    try {
      const res = await api("/api/users", { method: "POST", body: { username: name } });
      state.username = name;
      localStorage.setItem("glovesense.username", name);
      updateUserPill();
      if (res.created) {
        setStatus(el("setupStatus"), `Created ${res.path}`, "success");
        toast(`Profile ${name} created`, "success");
      } else {
        setStatus(el("setupStatus"), `Switched to existing profile.`);
      }
      state.cookieCount = 0;
      await refreshUsers();
      await refreshGestureGrid();
    } catch (e) {
      setStatus(el("setupStatus"), `Failed: ${e.message}`, "error");
    }
  });

  el("btnNewUser").addEventListener("click", () => {
    state.username = "";
    state.cookieCount = 0;
    state.knownGestures = new Set();
    localStorage.removeItem("glovesense.username");
    el("username").value = "";
    updateUserPill();
    resetCharts();
    setGestureImage(0);
    setStatus(el("setupStatus"), "Cleared current profile.");
    renderGestureGrid();
    refreshUsers();
  });
}

function wireVariantToggle() {
  el("variantToggle").addEventListener("click", (e) => {
    const btn = e.target.closest("button[data-variant]");
    if (!btn) return;
    if (state.ws) {
      toast("Finish or abort the current recording before switching profiles.", "warn");
      return;
    }
    const next = btn.dataset.variant;
    if (next === state.variant) return;
    state.variant = next;
    localStorage.setItem("glovesense.variant", next);
    renderVariantToggle();
    const target = VARIANT_TARGETS[next] ?? 105;
    if (!state.ws) setProgress(0, target);
    toast(`Switched to ${state.variants[next]?.label ?? next}`, "success");
  });
}

function wireTrain() {
  wireVariantToggle();
  el("btnNextGesture").addEventListener("click", () => {
    if (!ensureUser()) return;
    if (state.cookieCount >= 15) {
      setStatus(el("collectStatus"), "All 15 gestures recorded for this round.", "success");
      toast("Round complete!", "success");
      return;
    }
    state.cookieCount += 1;
    streamCollection(state.cookieCount, "train");
  });

  el("btnRepeatGesture").addEventListener("click", () => {
    if (!ensureUser()) return;
    streamCollection(Number(el("selRepeat").value), "train");
  });

  el("btnPause").addEventListener("click", togglePause);
  el("btnAbort").addEventListener("click", abortCollection);

  el("btnTrainModels").addEventListener("click", async () => {
    if (!ensureUser()) return;
    setStatus(el("trainStatus"), "Training every model (RF, LightGBM, SVM, k-NN, Ensemble)…", "warn");
    try {
      const result = await api(`/api/users/${encodeURIComponent(state.username)}/train`, {
        method: "POST",
        body: { feature_set: state.featureSet },
      });
      const top = (result.results || []).find((r) => r.train_accuracy != null);
      const note = top ? `Top: ${top.display} ${(top.train_accuracy * 100).toFixed(1)}%` : "Done.";
      setStatus(el("trainStatus"), `Trained ${result.results.length} models · ${note}`, "success");
      renderTrainResults(result);
      toast("Training complete", "success");
    } catch (e) {
      setStatus(el("trainStatus"), `Failed: ${e.message}`, "error");
      toast(`Training failed: ${e.message}`, "error");
    }
  });
}

function wireTest() {
  el("btnGetTestData").addEventListener("click", () => {
    if (!ensureUser()) return;
    clearReaction();
    streamCollection(Number(el("selTest").value), "test");
  });

  el("btnAbortTest").addEventListener("click", abortCollection);

  el("btnTestAccuracy").addEventListener("click", async () => {
    if (!ensureUser()) return;
    const n = Number(el("selTest").value);
    setStatus(el("testStatus"), `Predicting gesture ${state.mapping[n] ?? n} via ${displayModel(state.selectedModel)}…`, "warn");
    try {
      const result = await api(`/api/users/${encodeURIComponent(state.username)}/predict`, {
        method: "POST",
        body: { gesture_number: n, model: state.selectedModel, feature_set: state.featureSet },
      });
      const text = result.correct
        ? `✓ ${result.model_display}: ${result.predicted}`
        : `✗ ${result.model_display}: ${result.predicted}, expected ${result.expected}`;
      setStatus(el("testStatus"), text, result.correct ? "success" : "error");
      renderPrediction(result);
    } catch (e) {
      setStatus(el("testStatus"), `Failed: ${e.message}`, "error");
      toast(`Prediction failed: ${e.message}`, "error");
    }
  });

  const modelSel = el("selModel");
  if (modelSel) {
    modelSel.addEventListener("change", () => {
      state.selectedModel = modelSel.value;
      localStorage.setItem("glovesense.model", state.selectedModel);
    });
  }
}

function displayModel(key) {
  return state.models.find((m) => m.key === key)?.display ?? key;
}

async function checkHealth() {
  try {
    await api("/api/health");
    setPill(el("serverPill"), "● Online", "ok");
  } catch {
    setPill(el("serverPill"), "● Offline", "err");
  }
}

function setGlovePill(text, klass, title) {
  const pill = el("glovePill");
  if (!pill) return;
  pill.textContent = text;
  pill.classList.remove("glove-on", "glove-off", "glove-busy", "ok", "err", "active");
  pill.classList.add(klass);
  if (title) pill.title = title;
}

async function checkGloveStatus() {
  try {
    const s = await api("/api/glove/status");
    if (!s.connected) {
      const seen = (s.ports_seen || []).join(", ") || "none";
      setGlovePill(`🧤 ${s.port} disconnected`, "glove-off", `Configured port ${s.port} not present.\nPorts seen by Windows: ${seen}`);
      return;
    }
    const dev = s.device || {};
    const desc = dev.description || dev.manufacturer || dev.hwid || "device";
    if (s.busy) {
      setGlovePill(`🧤 ${s.port} streaming`, "glove-busy", `${desc}\nActive collection in progress.`);
    } else {
      setGlovePill(`🧤 ${s.port} ready`, "glove-on", `${desc}\n@ ${s.baud} baud`);
    }
  } catch {
    setGlovePill("🧤 Status unknown", "glove-off", "Backend did not respond.");
  }
}

function barClass(pct) {
  return pct >= 80 ? "" : pct >= 55 ? "mid" : "lo";
}

function leaderboardTakeaway(featureSet, results) {
  if (!results || results.length === 0) return "";
  const top = results[0];
  const second = results[1];
  if (top.error) return "";
  const gap = second && second.cv_accuracy_mean != null
    ? `${((top.cv_accuracy_mean - second.cv_accuracy_mean) * 100).toFixed(1)} pts ahead of ${second.display}`
    : "";
  const featLabel = featureSet === "engineered" ? "engineered 48-D" : "energy 8-D";
  return `<div class="takeaway"><strong>Best on the ${featLabel} pool:</strong> ${top.display} at ${(top.cv_accuracy_mean * 100).toFixed(1)}% (5-fold CV, ${gap}). This is your within-population estimate.</div>`;
}

function renderLeaderboard(d) {
  if (d.error) {
    el("leaderboardResults").innerHTML = `<p class="status error">${d.error}</p>`;
    return;
  }
  const rows = (d.results || [])
    .map((r, i) => {
      if (r.error) {
        return `<tr><td>${r.display}</td><td colspan="2" class="miss">${r.error}</td></tr>`;
      }
      const pct = r.cv_accuracy_mean * 100;
      return `
        <tr class="${i === 0 ? "winner" : ""}">
          <td>${r.display}</td>
          <td class="lb-bar-cell">
            <div class="lb-bar"><div class="lb-bar-fill ${barClass(pct)}" style="width:${pct.toFixed(1)}%"></div>
              <div class="lb-bar-label">${pct.toFixed(1)}%</div>
            </div>
          </td>
          <td>± ${(r.cv_accuracy_std * 100).toFixed(1)}%</td>
        </tr>`;
    })
    .join("");
  el("leaderboardResults").innerHTML = `
    <div class="lb-section-head">
      <h3>5-fold CV leaderboard</h3>
      <span class="meta">${d.samples} samples · ${d.users} users · ${d.feature_dim}-D</span>
    </div>
    <table class="lb-table"><thead><tr><th>Model</th><th>Mean accuracy</th><th>Std</th></tr></thead><tbody>${rows}</tbody></table>
    ${leaderboardTakeaway(d.feature_set, d.results)}
  `;
}

function renderLoou(d) {
  if (d.error) {
    el("loouResults").innerHTML = `<p class="status error">${d.error}</p>`;
    return;
  }
  const rows = (d.results || [])
    .map((r, i) => {
      const pct = (r.mean ?? 0) * 100;
      return `
        <tr class="${i === 0 ? "winner" : ""}">
          <td>${r.display}</td>
          <td class="lb-bar-cell">
            <div class="lb-bar"><div class="lb-bar-fill ${barClass(pct)}" style="width:${pct.toFixed(1)}%"></div>
              <div class="lb-bar-label">${pct.toFixed(1)}%</div>
            </div>
          </td>
          <td>± ${((r.std ?? 0) * 100).toFixed(1)}%</td>
        </tr>`;
    })
    .join("");
  const top = (d.results && d.results[0]) || null;
  const perUser = top
    ? top.per_user.map((u) => {
        const acc = u.accuracy != null ? u.accuracy : null;
        const pct = acc != null ? (acc * 100).toFixed(1) : "—";
        const klass = acc == null ? "" : acc >= 0.5 ? "good" : acc < 0.25 ? "bad" : "";
        return `<div class="lb-user-card ${klass}"><div class="name">${u.user}</div><div class="acc">${pct}%</div><div class="muted" style="font-size:0.7rem;">${u.samples} samples</div></div>`;
      }).join("")
    : "";

  const takeaway = top
    ? `<div class="takeaway"><strong>Generalization gap:</strong> the best model (${top.display}) averages ${(top.mean * 100).toFixed(1)}% leave-one-user-out vs. typically ~70%+ within-user CV. The current 8-D energy features encode user-specific hand shape more than the gesture itself — expect to retrain per user.</div>`
    : "";

  el("loouResults").innerHTML = `
    <div class="lb-section-head">
      <h3>Leave-one-user-out</h3>
      <span class="meta">${d.users} users · ${d.samples} samples</span>
    </div>
    <table class="lb-table"><thead><tr><th>Model</th><th>Mean accuracy across users</th><th>Std</th></tr></thead><tbody>${rows}</tbody></table>
    ${top ? `<div class="lb-section-head"><h3>Per-user accuracy (${top.display})</h3><span class="meta">held out one user at a time</span></div><div class="lb-user-grid">${perUser}</div>` : ""}
    ${takeaway}
  `;
}

function wireLab() {
  const featureToggle = el("featureToggle");
  if (featureToggle) {
    featureToggle.querySelectorAll("button").forEach((b) => b.classList.toggle("active", b.dataset.feature === state.featureSet));
    featureToggle.addEventListener("click", (e) => {
      const btn = e.target.closest("button[data-feature]");
      if (!btn) return;
      state.featureSet = btn.dataset.feature;
      localStorage.setItem("glovesense.featureSet", state.featureSet);
      featureToggle.querySelectorAll("button").forEach((b) => {
        b.classList.toggle("active", b === btn);
        b.setAttribute("aria-checked", b === btn ? "true" : "false");
      });
    });
  }

  const lbBtn = el("btnLeaderboard");
  const loouBtn = el("btnLoou");
  if (lbBtn) {
    lbBtn.addEventListener("click", async () => {
      setStatus(el("labStatus"), `Running 5-fold CV across every model on the ${state.featureSet} feature set… this can take ~30 s.`);
      lbBtn.disabled = true;
      loouBtn.disabled = true;
      try {
        const d = await api("/api/eval/leaderboard", { method: "POST", body: { feature_set: state.featureSet } });
        renderLeaderboard(d);
        setStatus(el("labStatus"), `Leaderboard ready (${d.samples} samples / ${d.users} users).`, "success");
      } catch (e) {
        setStatus(el("labStatus"), `Failed: ${e.message}`, "error");
      } finally {
        lbBtn.disabled = false;
        loouBtn.disabled = false;
      }
    });
  }
  if (loouBtn) {
    loouBtn.addEventListener("click", async () => {
      setStatus(el("labStatus"), `Running leave-one-user-out across every model… this can take a minute.`);
      lbBtn.disabled = true;
      loouBtn.disabled = true;
      try {
        const d = await api("/api/eval/loou", { method: "POST", body: { feature_set: state.featureSet } });
        renderLoou(d);
        setStatus(el("labStatus"), `LOOU ready (${d.users} users).`, "success");
      } catch (e) {
        setStatus(el("labStatus"), `Failed: ${e.message}`, "error");
      } finally {
        lbBtn.disabled = false;
        loouBtn.disabled = false;
      }
    });
  }
}

async function main() {
  initCharts();
  wireTabs();
  wireSetup();
  wireTrain();
  wireTest();
  wireLab();
  await loadGestureMapping();
  await loadModelCatalog();
  await refreshUsers();
  await refreshGestureGrid();
  setGestureImage(0);
  setPhase("idle", "Idle");
  setProgress(0, 0);
  await checkHealth();
  await checkGloveStatus();
  setInterval(checkHealth, 15000);
  setInterval(checkGloveStatus, 4000);
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) {
      checkHealth();
      checkGloveStatus();
    }
  });
}

window.addEventListener("DOMContentLoaded", main);
