(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);

  const dropzone = $("dropzone");
  const fileInput = $("fileInput");
  const demoBtn = $("demoBtn");

  const statusPill = $("statusPill");
  const statusDot = $("statusDot");
  const statusText = $("statusText");
  const elapsedClock = $("elapsedClock");

  const progStage = $("progStage");
  const progPct = $("progPct");
  const progFill = $("progFill");
  const liveFrame = $("liveFrame");
  const liveEmpty = $("liveEmpty");
  const consoleEl = $("console");

  const detectGrid = $("detectGrid");
  const detectEmpty = $("detectEmpty");
  const finalScores = $("finalScores");
  const metricsEl = $("metrics");
  const eventsWrap = $("eventsWrap");
  const eventsBody = $("eventsBody");

  const inputPreview = $("inputPreview");
  const inputVideo = $("inputVideo");
  const inputImage = $("inputImage");
  const inputSlideshow = $("inputSlideshow");
  const slideImg = $("slideImg");
  const slideLabel = $("slideLabel");
  const inputTag = $("inputTag");

  const lightbox = $("lightbox");
  const lightboxImg = $("lightboxImg");

  let poll = null;
  let elapsedTimer = null;
  let startTime = 0;
  let currentJob = null;
  let seenFrames = 0;
  let slideTimer = null;

  // ---------- helpers ----------
  function setStage(name) {
    const order = ["input", "run", "detect", "extract"];
    const idx = order.indexOf(name);
    order.forEach((s, i) => {
      const el = $("stage-" + s);
      if (!el) return;
      el.classList.remove("active", "done");
      if (i < idx) el.classList.add("done");
      else if (i === idx) el.classList.add("active");
    });
  }

  function setStatus(state, text) {
    statusPill.className = "status-pill " + state;
    statusText.textContent = text;
  }

  function log(message, cls = "") {
    const now = new Date();
    const t = now.toTimeString().slice(0, 8);
    const line = document.createElement("div");
    line.className = "line " + cls;
    line.innerHTML = `<span class="t">[${t}]</span> <span class="m"></span>`;
    line.querySelector(".m").textContent = " " + message;
    consoleEl.appendChild(line);
    consoleEl.scrollTop = consoleEl.scrollHeight;
  }

  function startClock() {
    startTime = Date.now();
    elapsedTimer = setInterval(() => {
      const s = (Date.now() - startTime) / 1000;
      elapsedClock.textContent = s.toFixed(2) + "s";
    }, 80);
  }
  function stopClock() { if (elapsedTimer) clearInterval(elapsedTimer); }

  function resetUI() {
    consoleEl.innerHTML = "";
    detectGrid.innerHTML = "";
    detectGrid.appendChild(detectEmpty);
    detectEmpty.style.display = "block";
    finalScores.innerHTML = '<div class="empty-note">Extracting…</div>';
    metricsEl.innerHTML = '<div class="empty-note">Running…</div>';
    eventsWrap.hidden = true;
    eventsBody.innerHTML = "";
    liveFrame.src = "";
    liveFrame.style.display = "none";
    liveEmpty.style.display = "block";
    progFill.style.width = "0%";
    progPct.textContent = "0%";
    progStage.textContent = "standby";
    seenFrames = 0;
  }

  // ---------- input previews ----------
  function showUploadPreview(file, kind) {
    inputPreview.hidden = false;
    inputVideo.hidden = true;
    inputImage.hidden = true;
    inputSlideshow.hidden = true;
    stopSlideshow();
    const url = URL.createObjectURL(file);
    inputTag.textContent = "INPUT · " + kind.toUpperCase();
    if (kind === "video") {
      inputVideo.src = url;
      inputVideo.hidden = false;
      inputVideo.play().catch(() => {});
    } else {
      inputImage.src = url;
      inputImage.hidden = false;
    }
  }

  async function showDemoPreview() {
    inputPreview.hidden = false;
    inputVideo.hidden = true;
    inputImage.hidden = true;
    inputSlideshow.hidden = false;
    inputTag.textContent = "INPUT · VIDEO STREAM";
    try {
      const r = await fetch("/api/demo-input");
      const data = await r.json();
      const frames = data.frames || [];
      if (!frames.length) return;
      let i = 0;
      slideImg.src = frames[0];
      slideLabel.textContent = frames[0].split("/").pop();
      stopSlideshow();
      slideTimer = setInterval(() => {
        i = (i + 1) % frames.length;
        slideImg.src = frames[i];
        slideLabel.textContent = frames[i].split("/").pop();
      }, 260);
    } catch (e) { /* ignore */ }
  }
  function stopSlideshow() { if (slideTimer) { clearInterval(slideTimer); slideTimer = null; } }

  // ---------- job lifecycle ----------
  async function startUpload(file) {
    const ext = file.name.split(".").pop().toLowerCase();
    const kind = ["mp4","mov","avi","mkv","webm","m4v"].includes(ext) ? "video" : "image";
    resetUI();
    showUploadPreview(file, kind);
    setStage("run");
    setStatus("running", "Uploading…");
    log("Uploading " + file.name + " …");

    const fd = new FormData();
    fd.append("file", file);
    try {
      const r = await fetch("/api/process", { method: "POST", body: fd });
      const data = await r.json();
      if (data.error) { fail(data.error); return; }
      log("Upload complete. Job " + data.job_id + " started.", "ok");
      beginPolling(data.job_id);
    } catch (e) { fail("Upload failed: " + e.message); }
  }

  async function startDemo() {
    resetUI();
    showDemoPreview();
    setStage("run");
    setStatus("running", "Starting demo…");
    log("Loading bundled bowling_scoreboard.mp4 frames…");
    try {
      const r = await fetch("/api/demo", { method: "POST" });
      const data = await r.json();
      if (data.error) { fail(data.error); return; }
      log("Demo job " + data.job_id + " started.", "ok");
      beginPolling(data.job_id);
    } catch (e) { fail("Demo failed: " + e.message); }
  }

  function beginPolling(jobId) {
    currentJob = jobId;
    startClock();
    if (poll) clearInterval(poll);
    poll = setInterval(() => tick(jobId), 350);
  }

  async function tick(jobId) {
    try {
      const r = await fetch("/api/status/" + jobId);
      const s = await r.json();
      if (s.error) return;

      if (typeof s.percent === "number") {
        progFill.style.width = s.percent + "%";
        progPct.textContent = s.percent + "%";
      }
      if (s.stage) progStage.textContent = s.stage;

      if (s.message && s.message !== window.__lastMsg) {
        window.__lastMsg = s.message;
        const cls = /confirm|complete|ready/i.test(s.message) ? "ok"
                  : /reject|error|fail/i.test(s.message) ? "err" : "";
        log(s.message, cls);
      }

      if (s.stage === "process" || s.stage === "analyze") setStage("detect");

      if (s.preview) {
        liveFrame.src = "/static/jobs/" + jobId + "/" + s.preview + "?t=" + Date.now();
        liveFrame.style.display = "block";
        liveEmpty.style.display = "none";
      }

      if (s.state === "complete" && s.result) {
        finish(jobId, s.result);
      } else if (s.state === "error") {
        fail(s.message || "Pipeline error.");
      }
    } catch (e) { /* transient */ }
  }

  function finish(jobId, result) {
    clearInterval(poll); poll = null;
    stopClock();
    setStatus("complete", "Complete");
    setStage("extract");
    $("stage-extract").classList.add("active");
    ["input","run","detect"].forEach(s => $("stage-"+s).classList.add("done"));
    log("Pipeline finished in " + result.metrics.elapsed_seconds + "s.", "ok");

    renderDetection(jobId, result.frames);
    renderFinalScores(result);
    renderMetrics(result.metrics);
    renderEvents(result.events);
    if (!result.metrics.ocr_available) {
      log("Note: OCR unavailable — showing localization results only.", "warn");
    }
  }

  function fail(msg) {
    if (poll) { clearInterval(poll); poll = null; }
    stopClock();
    setStatus("error", "Error");
    log(msg, "err");
  }

  // ---------- renderers ----------
  function renderDetection(jobId, frames) {
    detectGrid.innerHTML = "";
    const withBox = frames.filter(f => f.box);
    if (!withBox.length) {
      const n = document.createElement("div");
      n.className = "empty-note";
      n.textContent = "No scoreboard localized in the sampled frames.";
      detectGrid.appendChild(n);
      return;
    }
    withBox.forEach(f => {
      const cell = document.createElement("div");
      cell.className = "detect-cell";
      const src = "/static/jobs/" + jobId + "/" + f.frame_url;
      cell.innerHTML =
        `<img src="${src}" alt="Detected scoreboard in ${f.label}" loading="lazy" />
         <div class="cell-meta">
           <span class="st ${f.status}">${f.status}</span>
           <span>${(f.confidence * 100).toFixed(0)}%</span>
         </div>`;
      cell.addEventListener("click", () => openLightbox(src));
      detectGrid.appendChild(cell);
    });
  }

  function renderFinalScores(result) {
    const fs = result.final_scores || {};
    const players = result.players || Object.keys(fs);
    // build per-player net change from series
    const first = {}, last = {};
    (result.series || []).forEach(row => {
      players.forEach(p => {
        if (row[p] != null) { if (first[p] == null) first[p] = row[p]; last[p] = row[p]; }
      });
    });
    finalScores.innerHTML = "";
    const anyScore = players.some(p => fs[p] != null);
    if (!anyScore) {
      finalScores.innerHTML = '<div class="empty-note">No numeric scores were extracted from this input.</div>';
      return;
    }
    players.forEach((p, i) => {
      const v = fs[p];
      const delta = (first[p] != null && last[p] != null) ? last[p] - first[p] : 0;
      const card = document.createElement("div");
      card.className = "score-card" + (delta > 0 ? " gain" : "");
      const label = "P" + (i + 1);
      card.innerHTML =
        `<div class="pl">${label}</div>
         <div class="val">${v == null ? "—" : v}</div>
         ${delta > 0 ? `<div class="chg up">+${delta} this session</div>` : `<div class="chg">no change</div>`}`;
      finalScores.appendChild(card);
    });
  }

  function renderMetrics(m) {
    const items = [
      ["Frames processed", m.frames_processed, false],
      ["Localization coverage", m.localization_coverage.toFixed(1) + "%", true],
      ["Direct detections", m.direct_detections, false],
      ["Tracking holds", m.tracking_holds, false],
      ["Events detected", m.events_detected, false],
      ["Confirmed events", m.events_confirmed, true],
      ["Avg reliability", m.average_confirmed_reliability.toFixed(0) + "%", true],
      ["Total score gain", "+" + m.total_confirmed_increase, true],
    ];
    metricsEl.innerHTML = "";
    items.forEach(([k, v, accent]) => {
      const el = document.createElement("div");
      el.className = "metric";
      el.innerHTML = `<div class="k">${k}</div><div class="v ${accent ? "accent" : ""}">${v}</div>`;
      metricsEl.appendChild(el);
    });
  }

  function renderEvents(events) {
    if (!events || !events.length) { eventsWrap.hidden = true; return; }
    eventsWrap.hidden = false;
    eventsBody.innerHTML = "";
    events.forEach(e => {
      const tr = document.createElement("tr");
      const pl = "P" + (e.player.split("_")[1] || "?");
      const deltaTxt = (e.delta > 0 ? "+" : "") + e.delta;
      tr.innerHTML =
        `<td>${pl}</td>
         <td>${e.old_score} → ${e.new_score}</td>
         <td>${deltaTxt}</td>
         <td><span class="badge ${e.status}">${e.status}</span></td>
         <td>${e.reliability}%</td>
         <td>${e.event_frame}</td>`;
      eventsBody.appendChild(tr);
    });
  }

  // ---------- lightbox ----------
  function openLightbox(src) {
    lightboxImg.src = src;
    lightbox.hidden = false;
  }
  lightbox.addEventListener("click", () => { lightbox.hidden = true; lightboxImg.src = ""; });

  // ---------- events ----------
  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
  });
  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) startUpload(e.target.files[0]);
  });
  ["dragenter", "dragover"].forEach(ev =>
    dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add("drag"); }));
  ["dragleave", "drop"].forEach(ev =>
    dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove("drag"); }));
  dropzone.addEventListener("drop", (e) => {
    const f = e.dataTransfer.files && e.dataTransfer.files[0];
    if (f) startUpload(f);
  });
  demoBtn.addEventListener("click", startDemo);

  setStage("input");
  setStatus("", "Idle");
})();
