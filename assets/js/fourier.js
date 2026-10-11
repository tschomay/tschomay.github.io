/*
  Interactive widgets for the Fourier notes post.
  Three steps, each a <div class="fx" data-fx="..."> in the post:
    one     — a single sine wave with amplitude / frequency / phase sliders
    square  — match a square wave by hand-tuning 8 harmonics
    draw    — draw any shape, then rebuild it from the first K harmonics
    scope   — a synthetic powered-off scope trace hiding 60 Hz line noise
  Colors come from the site's CSS custom properties so light/dark both work.
*/
(function () {
  "use strict";

  const TAU = Math.PI * 2;

  function palette() {
    const s = getComputedStyle(document.documentElement);
    const v = (n) => s.getPropertyValue(n).trim();
    return {
      ink: v("--ink"), soft: v("--ink-soft"), line: v("--line"),
      signal: v("--signal"), teal: v("--teal"), tealSoft: v("--teal-soft"),
    };
  }

  // Size a canvas to its CSS box at device resolution; returns a ready ctx.
  function fit(canvas) {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    return { ctx, w, h };
  }

  function axis(g, c) {
    const { ctx, w, h } = g;
    ctx.strokeStyle = c.line;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, h / 2); ctx.lineTo(w, h / 2);
    ctx.stroke();
  }

  // Plot f(t) for t in [0, 1); yMax is the value that reaches the top edge.
  function plot(g, f, yMax, color, width, dash) {
    const { ctx, w, h } = g;
    const steps = Math.max(200, Math.round(w * 1.5));
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.lineJoin = "round";
    ctx.setLineDash(dash || []);
    ctx.beginPath();
    for (let i = 0; i <= steps; i++) {
      const t = i / steps;
      const y = h / 2 - (f(t) / yMax) * (h / 2 - 6);
      if (i === 0) ctx.moveTo(t * w, y); else ctx.lineTo(t * w, y);
    }
    ctx.stroke();
    ctx.setLineDash([]);
  }

  const fmt = (x, d) => (x < 0 ? "−" : "") + Math.abs(x).toFixed(d === undefined ? 2 : d);

  /* ---------- Step 1: one wave ---------- */
  function one(root) {
    const canvas = root.querySelector("canvas");
    const amp = root.querySelector('[name="amp"]');
    const freq = root.querySelector('[name="freq"]');
    const phase = root.querySelector('[name="phase"]');
    const eq = root.querySelector(".fx-eq");

    function draw() {
      const A = +amp.value, f = +freq.value, p = +phase.value;
      root.querySelector('[for="amp"]').textContent = fmt(A);
      root.querySelector('[for="freq"]').textContent = f + "×";
      root.querySelector('[for="phase"]').textContent = fmt(p / Math.PI) + "π";
      eq.textContent = "y = " + fmt(A) + " · sin(2π · " + f + "t + " + fmt(p / Math.PI) + "π)";
      const c = palette(), g = fit(canvas);
      axis(g, c);
      plot(g, (t) => A * Math.sin(TAU * f * t + p), 1.15, c.signal, 2.5);
    }
    [amp, freq, phase].forEach((el) => el.addEventListener("input", draw));
    return draw;
  }

  /* ---------- Step 2: build a square wave ---------- */
  function square(root) {
    const N = 8;
    const canvas = root.querySelector("canvas");
    const grid = root.querySelector(".fx-harmonics");
    const meter = root.querySelector(".fx-meter span");
    const score = root.querySelector(".fx-score");
    const note = root.querySelector(".fx-note");
    const showParts = root.querySelector('[name="parts"]');
    const amps = new Array(N).fill(0);
    const sliders = [];

    for (let n = 1; n <= N; n++) {
      const row = document.createElement("label");
      row.className = "fx-row";
      row.innerHTML =
        '<span class="fx-k">' + n + "×</span>" +
        '<input type="range" min="-1.3" max="1.3" step="0.01" value="0" aria-label="Amplitude of the ' + n + '× wave">' +
        '<output class="fx-v">0.00</output>';
      const input = row.querySelector("input");
      input.addEventListener("input", () => { amps[n - 1] = +input.value; draw(); });
      grid.appendChild(row);
      sliders.push(input);
    }

    const target = (t) => (t < 0.5 ? 1 : -1);
    const sum = (t) => amps.reduce((s, a, i) => s + a * Math.sin(TAU * (i + 1) * t), 0);

    // Share of the square wave's energy captured (1 = perfect, clamped at 0).
    function match() {
      const M = 1000;
      let err = 0;
      for (let i = 0; i < M; i++) {
        const t = (i + 0.5) / M, d = sum(t) - target(t);
        err += d * d;
      }
      return Math.max(0, 1 - err / M);
    }
    // Best you can do with these 8 harmonics: odd terms at 4/(πn), each holding 8/(πn)² of the energy.
    const bestPct = (8 / (Math.PI * Math.PI)) * [1, 3, 5, 7].reduce((s, n) => s + 1 / (n * n), 0);

    function setAll(values) {
      values.forEach((v, i) => { amps[i] = v; sliders[i].value = v; });
      draw();
    }
    root.querySelector('[data-act="solve"]').addEventListener("click", () =>
      setAll(amps.map((_, i) => ((i + 1) % 2 ? 4 / (Math.PI * (i + 1)) : 0))));
    root.querySelector('[data-act="reset"]').addEventListener("click", () => setAll(amps.map(() => 0)));
    showParts.addEventListener("change", () => draw());

    function draw() {
      sliders.forEach((s, i) => { s.nextElementSibling.textContent = fmt(amps[i]); });
      const c = palette(), g = fit(canvas);
      axis(g, c);
      plot(g, target, 1.7, c.soft, 1.5, [5, 5]);
      if (showParts.checked) {
        amps.forEach((a, i) => {
          if (Math.abs(a) > 0.005) plot(g, (t) => a * Math.sin(TAU * (i + 1) * t), 1.7, c.tealSoft, 1);
        });
      }
      plot(g, sum, 1.7, c.signal, 2.5);

      const m = match();
      meter.style.width = (m * 100).toFixed(1) + "%";
      score.textContent = Math.round(m * 100) + "% match";
      const evens = amps.some((a, i) => (i + 1) % 2 === 0 && Math.abs(a) > 0.08);
      if (m >= bestPct - 0.01) {
        note.textContent = "That’s as close as 8 sine waves can get (" + Math.round(bestPct * 100) +
          "%). The leftover wiggle lives in the 9×, 11×, 13×… waves — which is what step 3 is for.";
      } else if (m >= 0.85) {
        note.textContent = "Very close. Notice which sliders ended up mattering — and which ones you left at zero.";
      } else if (evens) {
        note.textContent = "Hint: a square wave looks the same flipped upside-down half a cycle later. Do the even harmonics?";
      } else if (amps[0] === 0) {
        note.textContent = "Start with 1× — the fundamental. How tall should it be to hug the square?";
      } else {
        note.textContent = "Now try adding a little of the higher harmonics to sharpen the corners.";
      }
    }
    return draw;
  }

  /* ---------- Step 3: draw anything ---------- */
  function drawAnything(root) {
    const S = 256, MAXK = 64, YMAX = 1.4;
    const canvas = root.querySelector(".fx-pad");
    const spec = root.querySelector(".fx-spec");
    const kIn = root.querySelector('[name="terms"]');
    const kOut = root.querySelector('[for="terms"]');
    const readout = root.querySelector(".fx-score");
    const data = new Float32Array(S);
    let re = new Float32Array(MAXK + 1), im = new Float32Array(MAXK + 1);

    const presets = {
      square: (t) => (t < 0.5 ? 1 : -1),
      saw: (t) => 1 - 2 * t,
      triangle: (t) => 1 - 4 * Math.abs(t - 0.5),
      heartbeat: (t) => {
        const bump = (c, w, a) => a * Math.exp(-((t - c) * (t - c)) / (2 * w * w));
        return -0.35 + bump(0.18, 0.03, 0.25) + bump(0.36, 0.008, -0.35) +
          bump(0.4, 0.012, 1.45) + bump(0.44, 0.008, -0.5) + bump(0.68, 0.05, 0.4);
      },
    };
    function load(name) {
      for (let i = 0; i < S; i++) data[i] = presets[name](i / S);
      transform(); draw();
    }

    // Plain DFT over the first MAXK harmonics — 256×64 is cheap enough per frame.
    function transform() {
      for (let k = 0; k <= MAXK; k++) {
        let r = 0, m = 0;
        for (let j = 0; j < S; j++) {
          const a = (TAU * k * j) / S;
          r += data[j] * Math.cos(a);
          m -= data[j] * Math.sin(a);
        }
        const scale = k === 0 ? 1 / S : 2 / S;
        re[k] = r * scale; im[k] = m * scale;
      }
    }
    function rebuild(t, K) {
      let y = re[0];
      for (let k = 1; k <= K; k++) y += re[k] * Math.cos(TAU * k * t) - im[k] * Math.sin(TAU * k * t);
      return y;
    }
    const sample = (t) => data[Math.min(S - 1, Math.floor(t * S))];

    function draw() {
      const K = +kIn.value;
      kOut.textContent = K;
      const c = palette();

      const g = fit(canvas);
      axis(g, c);
      plot(g, sample, YMAX, c.soft, 1.5, [5, 5]);
      plot(g, (t) => rebuild(t, K), YMAX, c.signal, 2.5);

      // Spectrum: one bar per harmonic, lit if it's in the rebuild.
      const s = fit(spec), mags = [];
      let peak = 1e-6, total = 0, used = 0;
      for (let k = 1; k <= MAXK; k++) {
        const m = Math.hypot(re[k], im[k]);
        mags.push(m); peak = Math.max(peak, m);
        total += m * m; if (k <= K) used += m * m;
      }
      const bw = s.w / MAXK;
      mags.forEach((m, i) => {
        const bh = (m / peak) * (s.h - 4);
        s.ctx.fillStyle = i < K ? c.signal : c.line;
        s.ctx.fillRect(i * bw + 1, s.h - bh, Math.max(1, bw - 2), bh);
      });
      readout.textContent = total > 1e-9
        ? K + (K === 1 ? " wave" : " waves") + " · " + Math.min(100, (used / total) * 100).toFixed(1) + "% of the shape"
        : "Draw something above";
    }

    // Drawing: each pointer move sets the samples it sweeps across.
    let last = null;
    function at(e) {
      const r = canvas.getBoundingClientRect();
      const i = Math.max(0, Math.min(S - 1, Math.floor(((e.clientX - r.left) / r.width) * S)));
      const y = Math.max(-YMAX, Math.min(YMAX, (1 - (e.clientY - r.top) / (r.height / 2)) * YMAX));
      return [i, y];
    }
    function paint(e) {
      const [i, y] = at(e);
      if (last) {
        const [i0, y0] = last, lo = Math.min(i0, i), hi = Math.max(i0, i);
        for (let j = lo; j <= hi; j++) data[j] = hi === lo ? y : y0 + ((y - y0) * (j - i0)) / (i - i0);
      } else data[i] = y;
      last = [i, y];
      transform(); draw();
    }
    let down = false;
    canvas.addEventListener("pointerdown", (e) => {
      e.preventDefault();
      down = true; last = null;
      try { canvas.setPointerCapture(e.pointerId); } catch (_) { /* capture is a nicety */ }
      paint(e);
    });
    canvas.addEventListener("pointermove", (e) => { if (down) paint(e); });
    const stop = () => { down = false; last = null; };
    canvas.addEventListener("pointerup", stop);
    canvas.addEventListener("pointercancel", stop);

    root.querySelectorAll("[data-preset]").forEach((b) =>
      b.addEventListener("click", () => load(b.dataset.preset)));
    kIn.addEventListener("input", draw);
    load("heartbeat");
    return draw;
  }

  /* ---------- Step 4: the scope with the power off ---------- */
  function scope(root) {
    const FS = 5000, N = 1000, BINS = 200; // 0.2 s at 5 kHz → 5 Hz bins, show 0–1000 Hz
    const SHOW = N / 2;                    // plot the first 0.1 s
    const trace = root.querySelector(".fx-plot");
    const spec = root.querySelector(".fx-scope-spec");
    const reveal = root.querySelector(".fx-reveal");
    const rmsOut = root.querySelector(".fx-rms");
    const note = root.querySelector(".fx-note");

    // Synthetic "powered off" measurement (mV): mains hum, a few harmonics,
    // and broadband noise. Seeded so every reader sees the same trace.
    let seed = 7;
    const rand = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
    const gauss = () => Math.sqrt(-2 * Math.log(rand() + 1e-12)) * Math.cos(TAU * rand());
    const hum = [[60, 1.0, 0.3], [120, 0.06, 1.1], [180, 0.35, 2.0], [300, 0.18, 0.7], [420, 0.1, 4.0], [540, 0.05, 1.5]];
    const raw = new Float32Array(N);
    for (let j = 0; j < N; j++) {
      const t = j / FS;
      let y = 0;
      hum.forEach(([f, a, p]) => { y += a * Math.sin(TAU * f * t + p); });
      raw[j] = y + 0.55 * gauss();
    }

    // Real DFT of every bin we display.
    const re = new Float32Array(BINS + 1), im = new Float32Array(BINS + 1), mag = new Float32Array(BINS + 1);
    for (let k = 1; k <= BINS; k++) {
      let r = 0, m = 0;
      for (let j = 0; j < N; j++) {
        const a = (TAU * k * j) / N;
        r += raw[j] * Math.cos(a); m -= raw[j] * Math.sin(a);
      }
      re[k] = (2 * r) / N; im[k] = (2 * m) / N; mag[k] = Math.hypot(re[k], im[k]);
    }
    const hz = (k) => (k * FS) / N;
    const removed = new Set();
    const humBins = hum.map(([f]) => Math.round((f * N) / FS));

    // Cleaned trace = raw minus whichever bins have been filtered out.
    function cleaned(j) {
      let y = raw[j];
      removed.forEach((k) => { y -= re[k] * Math.cos((TAU * k * j) / N) - im[k] * Math.sin((TAU * k * j) / N); });
      return y;
    }
    const rms = (f) => { let s = 0; for (let j = 0; j < N; j++) s += f(j) ** 2; return Math.sqrt(s / N); };
    const rms0 = rms((j) => raw[j]);

    function draw() {
      const c = palette();
      const g = fit(trace);
      axis(g, c);
      const at = (t) => cleaned(Math.min(N - 1, Math.floor(t * SHOW)));
      if (removed.size) plot(g, (t) => raw[Math.min(N - 1, Math.floor(t * SHOW))], 3.2, c.line, 1);
      plot(g, at, 3.2, removed.size ? c.signal : c.soft, 1.5);
      const now = removed.size ? rms(cleaned) : rms0;
      rmsOut.textContent = removed.size
        ? "noise " + rms0.toFixed(2) + " → " + now.toFixed(2) + " mV rms"
        : "noise " + rms0.toFixed(2) + " mV rms";

      if (reveal.hidden) return;
      const s = fit(spec), bw = s.w / BINS, top = 16;
      const peak = Math.max(...mag);
      for (let k = 1; k <= BINS; k++) {
        const bh = (mag[k] / peak) * (s.h - top - 14);
        s.ctx.fillStyle = removed.has(k) ? c.line : humBins.includes(k) ? c.signal : c.tealSoft;
        s.ctx.fillRect((k - 1) * bw, s.h - 14 - bh, Math.max(1, bw - 1), bh);
      }
      s.ctx.font = "10px " + getComputedStyle(root).fontFamily;
      s.ctx.fillStyle = c.soft;
      s.ctx.textAlign = "center";
      [60, 180, 300, 420].forEach((f) => {
        const k = Math.round((f * N) / FS);
        if (removed.has(k)) return;
        const bh = (mag[k] / peak) * (s.h - top - 14);
        s.ctx.fillText(f + " Hz", (k - 0.5) * bw, s.h - 14 - bh - 4);
      });
      [0, 250, 500, 750, 1000].forEach((f) => {
        s.ctx.textAlign = f === 0 ? "left" : f === 1000 ? "right" : "center";
        s.ctx.fillText(f, (f / hz(BINS)) * s.w, s.h - 2);
      });

      const left = humBins.filter((k) => !removed.has(k)).length;
      if (!removed.size) {
        note.textContent = "That’s not random. A spike at exactly 60 Hz, then 180, 300, 420… Try tapping the tallest one.";
      } else if (left) {
        note.textContent = "The trace got quieter. " + left + " of the 60 Hz family still left — the brass bars.";
      } else {
        note.textContent = "Hum gone. What’s left is the flat, broadband noise floor — the stuff no single frequency explains.";
      }
    }

    // Tap a bar → remove the tallest bin within a few bins of the tap (snap to the peak).
    spec.addEventListener("click", (e) => {
      const r = spec.getBoundingClientRect();
      const k0 = Math.floor(((e.clientX - r.left) / r.width) * BINS) + 1;
      let best = k0;
      for (let k = Math.max(1, k0 - 3); k <= Math.min(BINS, k0 + 3); k++) if (mag[k] > mag[best]) best = k;
      if (removed.has(best)) removed.delete(best); else removed.add(best);
      draw();
    });
    root.querySelector('[data-act="fourier"]').addEventListener("click", (e) => {
      reveal.hidden = false; e.currentTarget.parentElement.hidden = true; draw();
    });
    root.querySelector('[data-act="hum"]').addEventListener("click", () => { humBins.forEach((k) => removed.add(k)); draw(); });
    root.querySelector('[data-act="restore"]').addEventListener("click", () => { removed.clear(); draw(); });
    return draw;
  }

  // Every slider: double-tap (or double-click) resets it to its starting value,
  // and while dragging, fine-grained sliders snap to that value when close.
  function homeable(input) {
    const home = +input.defaultValue;
    const min = +input.min, max = +input.max, step = +input.step || 1;
    const fine = (max - min) / step >= 100;
    const snap = 0.03 * (max - min);
    let dragging = false, lastDown = -Infinity, resetOnUp = false;
    const set = (v) => { input.value = v; input.dispatchEvent(new Event("input", { bubbles: true })); };
    input.addEventListener("pointerdown", (e) => {
      dragging = true;
      resetOnUp = e.timeStamp - lastDown < 350;
      lastDown = e.timeStamp;
    });
    const up = () => {
      if (!dragging) return;
      dragging = false;
      // Wait a tick so the browser's own thumb move lands first, then override it.
      if (resetOnUp) setTimeout(() => set(home), 0);
      resetOnUp = false;
    };
    window.addEventListener("pointerup", up);
    window.addEventListener("pointercancel", up);
    // Snap only while dragging, so arrow keys can still step away from home.
    input.addEventListener("input", (e) => {
      if (!dragging || !fine || !e.isTrusted) return;
      if (+input.value !== home && Math.abs(+input.value - home) < snap) set(home);
    });
  }

  const kinds = { one, square, draw: drawAnything, scope };
  const redraws = [];
  document.querySelectorAll(".fx[data-fx]").forEach((el) => {
    const make = kinds[el.dataset.fx];
    if (make) { const r = make(el); redraws.push(r); r(); }
    el.querySelectorAll('input[type="range"]').forEach(homeable);
  });
  let pending = 0;
  const all = () => { cancelAnimationFrame(pending); pending = requestAnimationFrame(() => redraws.forEach((r) => r())); };
  window.addEventListener("resize", all);
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", all);
})();
