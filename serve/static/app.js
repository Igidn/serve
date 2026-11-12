"use strict";

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

const ui = {
  snapshot: null,
  filter: "all",
  logSource: "engine",
  logTimer: null,
  stateTimer: null,
  idleFocused: false,
};

const esc = (value) =>
  String(value ?? "").replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  })[ch]);

const fmtBytes = (bytes) => {
  if (!bytes) return "0 B";
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  let size = Number(bytes);
  let i = 0;
  while (size >= 1024 && i < units.length - 1) {
    size /= 1024;
    i += 1;
  }
  return `${i === 0 ? Math.round(size) : size.toFixed(1)} ${units[i]}`;
};

const fmtDuration = (seconds) => {
  if (seconds == null) return "—";
  const s = Math.max(0, Math.floor(seconds));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  if (h) return `${h}h ${String(m).padStart(2, "0")}m`;
  return `${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}`;
};

const fmtClock = (epoch) => {
  if (!epoch) return "—";
  return new Date(epoch * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
};

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "content-type": "application/json" },
    ...options,
    body: options.body ? JSON.stringify(options.body) : undefined,
  });
  let payload = null;
  try {
    payload = await response.json();
  } catch (_) {
    payload = null;
  }
  if (!response.ok) {
    const message = payload?.error?.message || payload?.error || `request failed (${response.status})`;
    throw new Error(message);
  }
  return payload;
}

let toastTimer = null;
function toast(message, isError = false) {
  const node = $("#toast");
  node.textContent = message;
  node.classList.toggle("err", isError);
  node.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => node.classList.remove("show"), isError ? 6000 : 2800);
}

async function copy(text, label = "copied") {
  try {
    await navigator.clipboard.writeText(text);
    toast(label);
  } catch (_) {
    toast("clipboard unavailable — select the text instead", true);
  }
}

/* ---------------------------------------------------------------- actions */

async function run(action, success) {
  try {
    await action();
    if (success) toast(success);
    await refreshState();
  } catch (error) {
    toast(error.message, true);
  }
}

function loadModel(model) {
  const snapshot = ui.snapshot || {};
  const active = snapshot.active;
  const failed = active && active.key === model.key && snapshot.phase === "error";
  const replace = Boolean(active && (active.key !== model.key || failed));
  const warning = failed
    ? `Retry loading ${model.key}? The previous attempt failed; check the logs.`
    : active && active.key !== model.key
      ? `Unload ${active.key} and load ${model.key} instead?`
      : `Load ${model.key}? The weights must already be on disk.`;
  if (!confirm(warning)) return;
  run(() => api(`/api/models/${encodeURIComponent(model.key)}/load`, {
    method: "POST",
    body: { replace },
  }));
}

function downloadModel(model) {
  run(
    () => api(`/api/models/${encodeURIComponent(model.key)}/download`, { method: "POST" }),
    `downloading ${model.key} in the background`
  );
}

function deleteModel(model) {
  if (!confirm(`Delete the cached files for ${model.key}?`)) return;
  run(
    () => api(`/api/models/${encodeURIComponent(model.key)}/cache`, { method: "DELETE" }),
    "cache deleted"
  );
}

function removeModel(model) {
  if (!confirm(`Remove ${model.key} from the model list? Weights on disk are kept.`)) return;
  run(() => api(`/api/models/${encodeURIComponent(model.key)}`, { method: "DELETE" }), "removed");
}

function unloadModel() {
  run(() => api("/api/models/unload", { method: "POST" }), "engine stopped");
}

function shutdown() {
  if (!confirm("End this Kaggle run now? The engine stops and the GPU session closes.")) return;
  run(() => api("/api/shutdown", { method: "POST" }), "ending the session…");
}

/* ---------------------------------------------------------------- render */

function renderTunnel(snapshot) {
  const tunnel = snapshot.tunnel || {};
  const pill = $("#tunnel");
  const text = $("#tunnel-text");
  pill.classList.remove("ok", "err", "busy");
  if (tunnel.state === "ready" && tunnel.url) {
    pill.classList.add("ok");
    pill.href = tunnel.url;
    text.textContent = tunnel.url.replace(/^https:\/\//, "");
    pill.title = `${tunnel.url} — open in a new tab`;
  } else if (tunnel.state === "error") {
    pill.classList.add("err");
    text.textContent = "tunnel error — retry";
    pill.href = "#";
    pill.onclick = (event) => {
      event.preventDefault();
      run(() => api("/api/tunnel/retry", { method: "POST" }), "retrying the tunnel");
    };
  } else if (tunnel.state === "disabled") {
    text.textContent = "tunnel off";
    pill.href = "#";
  } else {
    pill.classList.add("busy");
    text.textContent = "opening tunnel…";
    pill.href = "#";
  }
  $("#identity").disabled = !tunnel.has_identity;
}

function renderGpus(snapshot) {
  const host = $("#gpus");
  const gpus = snapshot.gpus || [];
  if (!gpus.length) {
    host.innerHTML = `<div class="gpu"><b>no GPU</b>local mode</div>`;
    return;
  }
  host.innerHTML = gpus.map((gpu) => {
    const pct = gpu.memory_total_mb ? (gpu.memory_used_mb / gpu.memory_total_mb) * 100 : 0;
    return `<div class="gpu" title="${esc(gpu.name)}">
      <b>${esc(gpu.name.replace(/NVIDIA /, ""))}</b>
      ${(gpu.memory_used_mb / 1024).toFixed(1)}/${(gpu.memory_total_mb / 1024).toFixed(1)} GB · ${gpu.utilization}%
      <div class="bar"><span style="width:${pct.toFixed(1)}%"></span></div>
    </div>`;
  }).join("");
}

function renderIdle(snapshot) {
  const node = $("#idle");
  node.textContent = snapshot.idle_paused ? "paused" : fmtDuration(snapshot.idle_remaining);
  node.parentElement.classList.toggle("paused", Boolean(snapshot.idle_paused));
}

function modelTags(model, snapshot) {
  const tags = [];
  if (model.local_path) tags.push(`<span class="tag ok">kaggle input</span>`);
  else if (model.cached) tags.push(`<span class="tag ok">on disk ${fmtBytes(model.disk_bytes)}</span>`);
  else tags.push(`<span class="tag">not downloaded · ${model.size_gb ? `~${model.size_gb} GB` : "size ?"}</span>`);

  if (model.job && model.job.state === "running") {
    const pct = model.job.bytes_total
      ? ` ${Math.floor((model.job.bytes_done / model.job.bytes_total) * 100)}%`
      : "";
    tags.push(`<span class="tag warn">downloading${pct}</span>`);
  }
  if (model.job && model.job.state === "error") tags.push(`<span class="tag err">download failed</span>`);
  if (model.tensor_parallel > 1) tags.push(`<span class="tag">tp=${model.tensor_parallel}</span>`);
  if (model.engine === "vllm-omni") tags.push(`<span class="tag">omni</span>`);
  if (snapshot.vram_total_gb > 0 && model.min_vram_gb > snapshot.vram_total_gb) {
    tags.push(`<span class="tag warn">needs ~${model.min_vram_gb} GB vram</span>`);
  }
  if (snapshot.disk && !model.available && !model.job) {
    const freeGb = snapshot.disk.free / 1024 ** 3;
    if (model.size_gb && freeGb < model.size_gb * 1.12 + 1) {
      tags.push(`<span class="tag err">wont fit on working disk</span>`);
    }
  }
  return tags.join("");
}

function renderModels(snapshot) {
  const host = $("#model-list");
  const active = snapshot.active;
  const busy = snapshot.phase === "loading" || snapshot.phase === "stopping";
  const models = (snapshot.models || []).filter(
    (model) => ui.filter === "all" || model.kind === ui.filter
  );

  if (!models.length) {
    host.innerHTML = `<div class="empty">No models in this category.</div>`;
    return;
  }

  host.innerHTML = models.map((model) => {
    const isActive = active && active.key === model.key && snapshot.phase === "ready";
    const isLoading = active && active.key === model.key && snapshot.phase === "loading";
    const isFailed = active && active.key === model.key && snapshot.phase === "error";
    const actions = [];

    if (isLoading) {
      actions.push(`<button class="ghost small grow" disabled>loading…</button>`);
    } else if (isActive) {
      actions.push(`<button class="ghost small grow" data-action="unload" data-key="${esc(model.key)}">unload</button>`);
    } else if (isFailed) {
      actions.push(`<button class="primary small grow" data-action="load" data-key="${esc(model.key)}">retry</button>`);
      actions.push(`<button class="ghost tiny" data-action="unload" data-key="${esc(model.key)}">clear</button>`);
    } else if (!model.available) {
      actions.push(`<button class="primary small grow" data-action="download" data-key="${esc(model.key)}" ${busy ? "disabled" : ""}>download</button>`);
    } else {
      actions.push(`<button class="primary small grow" data-action="load" data-key="${esc(model.key)}" ${busy ? "disabled" : ""}>load</button>`);
    }
    if (model.cached && !isActive) {
      actions.push(`<button class="ghost tiny" data-action="delete" data-key="${esc(model.key)}" title="delete cached weights">del</button>`);
    }
    if (!model.builtin) {
      actions.push(`<button class="ghost tiny" data-action="remove" data-key="${esc(model.key)}" title="remove from the list">×</button>`);
    }

    return `<article class="model ${isActive ? "active" : ""}">
      <div class="model-top">
        <span class="model-name">${esc(model.label)}</span>
        <span class="model-kind">${esc(model.kind)}</span>
      </div>
      <div class="model-repo">${esc(model.repo)}</div>
      <div class="model-meta">${modelTags(model, snapshot)}</div>
      ${model.notes ? `<div class="model-note">${esc(model.notes)}</div>` : ""}
      <div class="model-actions">${actions.join("")}</div>
    </article>`;
  }).join("");
}

function endpointRow(label, value) {
  return `<div class="endpoint">
    <span class="micro">${esc(label)}</span>
    <code>${esc(value)}</code>
    <button class="ghost tiny" data-copy="${esc(value)}">copy</button>
  </div>`;
}

function renderActive(snapshot) {
  const host = $("#active-body");
  const phase = snapshot.phase;
  const badge = $("#phase");
  badge.textContent = snapshot.stage || phase;
  badge.className = `badge ${
    phase === "ready" ? "ready" : phase === "error" ? "error" : phase === "idle" ? "" : "busy"
  }`;

  const progress = $("#progress");
  if (phase === "loading") {
    progress.hidden = false;
    const pct = snapshot.progress;
    progress.classList.toggle("indeterminate", pct == null);
    $(".bar span", progress).style.width = pct == null ? "" : `${(pct * 100).toFixed(1)}%`;
    const left = snapshot.bytes_total
      ? `${fmtBytes(snapshot.bytes_done)} / ${fmtBytes(snapshot.bytes_total)}`
      : snapshot.bytes_done
        ? fmtBytes(snapshot.bytes_done)
        : "";
    $(".progress-label", progress).innerHTML = `<span>${esc(snapshot.stage || "working")}</span><span>${esc(left)}</span>`;
  } else {
    progress.hidden = true;
  }

  const active = snapshot.active;
  if (!active) {
    host.innerHTML = `<div class="active-none">
      Nothing is loaded. Pick a model on the left; the dashboard downloads it if
      needed, starts vLLM, then forwards <code>/v1/*</code> to it.
    </div>
    ${snapshot.error ? `<div class="error-box">${esc(snapshot.error)}</div>` : ""}`;
    return;
  }

  const base = `${location.origin}/v1`;
  const model = (snapshot.models || []).find((m) => m.key === active.key) || {};
  const uptime = snapshot.started_at && phase === "ready" ? (Date.now() / 1000 - snapshot.started_at) : null;
  const sample = active.kind === "image"
    ? `curl ${base}/images/generations \\
  -H 'content-type: application/json' \\
  -d '{
    "model": "${active.served_name}",
    "prompt": "a ceramic teapot on a wooden table",
    "size": "1024x1024",
    "num_inference_steps": 20
  }'`
    : `curl ${base}/chat/completions \\
  -H 'content-type: application/json' \\
  -d '{
    "model": "${active.served_name}",
    "messages": [{"role": "user", "content": "hello"}],
    "stream": true
  }'`;

  host.innerHTML = `
    <div class="active-title">
      <h3>${esc(active.label)}</h3>
      <span class="tag ok">${esc(active.engine)}</span>
      <span class="tag">${esc(active.kind)}</span>
    </div>
    <div class="active-sub">${esc(active.repo)} · ${esc(active.source || "")}</div>

    <div class="stat-grid">
      <div class="stat"><span class="micro">model name</span><b>${esc(active.served_name)}</b></div>
      <div class="stat"><span class="micro">weights</span><b>${esc(model.local_path ? "kaggle input" : fmtBytes(model.disk_bytes))}</b></div>
      <div class="stat"><span class="micro">uptime</span><b>${esc(uptime == null ? "starting" : fmtDuration(uptime))}</b></div>
      <div class="stat"><span class="micro">loaded at</span><b>${esc(fmtClock(snapshot.started_at))}</b></div>
    </div>

    ${endpointRow("base", `${base}`)}
    ${endpointRow(active.kind === "image" ? "images" : "chat", `${base}${active.kind === "image" ? "/images/generations" : "/chat/completions"}`)}
    ${endpointRow("models", `${base}/models`)}

    <div class="snippet">
      <pre>${esc(sample)}</pre>
      <button class="ghost tiny" data-copy="${esc(sample)}">copy</button>
    </div>

    ${snapshot.error ? `<div class="error-box">${esc(snapshot.error)}</div>` : ""}
  `;
}

function renderRuntime(snapshot) {
  const host = $("#runtime-body");
  const items = snapshot.runtime || [];
  host.innerHTML = items.map((item) => {
    const status = item.installing
      ? `<span class="status busy">installing…</span>`
      : item.installed
        ? `<span class="status ok">ready</span>`
        : `<span class="status err">missing</span>`;
    const button = !item.installed && !item.installing
      ? `<button class="ghost tiny" data-install="${esc(item.engine)}">install</button>`
      : "";
    const error = item.error ? `<div class="error-box">${esc(item.error)}</div>` : "";
    return `<div class="rt">
        <span class="name">${esc(item.engine)}</span>
        ${status}
        ${button}
        <span class="ver">${esc(item.version || "")}</span>
      </div>${error}`;
  }).join("") + `<div class="kv"><span class="k">installer log</span><span class="v">install tab</span></div>`;
}

function renderSession(snapshot) {
  const disk = snapshot.disk || {};
  const usedPct = disk.total ? (disk.used / disk.total) * 100 : 0;
  $("#session-body").innerHTML = `
    <div class="kv"><span class="k">root</span><span class="v">${esc(snapshot.app?.root || "")}</span></div>
    <div class="kv"><span class="k">disk free</span><span class="v">${fmtBytes(disk.free)} / ${fmtBytes(disk.total)}</span></div>
    <div class="kv"><span class="k">tunnel</span><span class="v">${esc(snapshot.tunnel?.url || snapshot.tunnel?.state || "")}</span></div>
    <div class="kv"><span class="k">exit on idle</span><span class="v">${snapshot.app?.allow_exit ? "kernel ends" : "disabled locally"}</span></div>
    <div class="gpu" style="margin-top:8px"><b>working disk</b>${usedPct.toFixed(0)}% used
      <div class="bar"><span style="width:${Math.min(100, usedPct).toFixed(1)}%;background:${usedPct > 88 ? "var(--danger)" : "var(--accent)"}"></span></div>
    </div>`;
  if (!ui.idleFocused) {
    const input = $("#idle-input");
    if (document.activeElement !== input) input.value = Math.round((snapshot.idle_timeout || 600) / 60);
  }
}

function render(snapshot) {
  ui.snapshot = snapshot;
  renderTunnel(snapshot);
  renderGpus(snapshot);
  renderIdle(snapshot);
  renderModels(snapshot);
  renderActive(snapshot);
  renderRuntime(snapshot);
  renderSession(snapshot);
}

/* ---------------------------------------------------------------- polling */

async function refreshState() {
  try {
    render(await api("/api/state"));
  } catch (error) {
    $("#tunnel-text").textContent = "dashboard unreachable";
  }
}

async function refreshLogs() {
  if (document.visibilityState !== "visible") return;
  const source = ui.logSource;
  try {
    const payload = await api(`/api/logs?source=${encodeURIComponent(source)}&lines=300`);
    const pre = $("#log");
    if (payload.text !== pre.textContent) {
      pre.textContent = payload.text || "(no output yet)";
      if ($("#follow").checked) pre.scrollTop = pre.scrollHeight;
    }
  } catch (_) {
    /* transient while the app restarts */
  }
}

function pollState() {
  clearTimeout(ui.stateTimer);
  refreshState().finally(() => {
    ui.stateTimer = setTimeout(pollState, 2000);
  });
}

function pollLogs() {
  clearTimeout(ui.logTimer);
  refreshLogs().finally(() => {
    ui.logTimer = setTimeout(pollLogs, 2500);
  });
}

/* ---------------------------------------------------------------- events */

document.addEventListener("click", (event) => {
  const copyButton = event.target.closest("[data-copy]");
  if (copyButton) {
    copy(copyButton.dataset.copy, "copied to clipboard");
    return;
  }

  const actionButton = event.target.closest("[data-action]");
  if (actionButton) {
    const model = (ui.snapshot?.models || []).find((m) => m.key === actionButton.dataset.key);
    if (!model) return;
    const action = actionButton.dataset.action;
    if (action === "load") loadModel(model);
    if (action === "download") downloadModel(model);
    if (action === "delete") deleteModel(model);
    if (action === "remove") removeModel(model);
    if (action === "unload") unloadModel();
    return;
  }

  const install = event.target.closest("[data-install]");
  if (install) {
    run(
      () => api(`/api/runtime/${encodeURIComponent(install.dataset.install)}/install`, { method: "POST" }),
      "install started — watch the install log"
    );
  }
});

$("#filter").addEventListener("click", (event) => {
  const button = event.target.closest("button[data-filter]");
  if (!button) return;
  ui.filter = button.dataset.filter;
  $$("#filter button").forEach((node) => node.classList.toggle("on", node === button));
  if (ui.snapshot) renderModels(ui.snapshot);
});

$("#log-tabs").addEventListener("click", (event) => {
  const button = event.target.closest("button[data-source]");
  if (!button) return;
  ui.logSource = button.dataset.source;
  $$("#log-tabs button").forEach((node) => node.classList.toggle("on", node === button));
  $("#log").textContent = "loading…";
  refreshLogs();
});

$("#identity").addEventListener("click", async () => {
  try {
    const payload = await api("/api/tunnel/identity");
    await copy(payload.b64, "identity copied — add it as the OPENTUNNEL_IDENTITY_B64 secret");
  } catch (error) {
    toast(error.message, true);
  }
});

$("#power").addEventListener("click", shutdown);

$("#idle-input").addEventListener("focus", () => { ui.idleFocused = true; });
$("#idle-input").addEventListener("blur", () => { ui.idleFocused = false; });

$("#idle-save").addEventListener("click", () => {
  const minutes = Number($("#idle-input").value);
  if (!Number.isFinite(minutes) || minutes < 1) return toast("enter at least one minute", true);
  run(
    () => api("/api/settings", { method: "PUT", body: { idle_timeout: minutes * 60 } }),
    `idle timeout set to ${minutes} minutes`
  );
});

$("#add-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const form = new FormData(event.target);
  const repo = String(form.get("repo") || "").trim();
  const local = String(form.get("local_path") || "").trim();
  const extras = String(form.get("extra_args") || "").trim();
  const maxLen = Number(form.get("max_model_len"));
  const payload = {
    repo,
    kind: form.get("kind"),
    engine: form.get("engine"),
    tensor_parallel: Number(form.get("tensor_parallel")) || 1,
    max_model_len: Number.isFinite(maxLen) && maxLen > 0 ? maxLen : null,
    extra_args: extras ? extras.split(/\s+/).filter(Boolean) : [],
    local_paths: local ? [local] : [],
  };
  run(async () => {
    await api("/api/models", { method: "POST", body: payload });
    event.target.reset();
    $("#add-model").open = false;
  }, "model added");
});

document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible") {
    refreshState();
    refreshLogs();
  }
});

pollState();
pollLogs();
