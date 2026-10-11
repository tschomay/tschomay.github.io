/*
  Interactive widgets for the Z-transform notes post.
  Each is a <div class="fx" data-fx="..."> in the post:
    spiral  — one z^n: radius and angle sliders, the z-plane and the sequence it makes
    poles   — drag a pole pair (and optionally a zero pair) around the z-plane;
              see the impulse response and the frequency response it produces
    notch   — a sample-by-sample 60 Hz notch filter run on a synthetic scope trace
  Reuses the .fx widget styles from fourier.css; colors come from the site's
  CSS custom properties so light/dark both work.
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

  // Lollipop plot of a sequence; values past ±yMax are clipped and marked hollow.
  function stems(g, c, vals, yMax, color) {
    const { ctx, w, h } = g;
    const dx = w / vals.length, half = h / 2 - 6;
    axis(g, c);
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 1.5;
    vals.forEach((v, i) => {
      const clipped = !isFinite(v) || Math.abs(v) > yMax;
      const vv = clipped ? (v < 0 ? -yMax : yMax) : v;
      const x = (i + 0.5) * dx, y = h / 2 - (vv / yMax) * half;
      ctx.beginPath(); ctx.moveTo(x, h / 2); ctx.lineTo(x, y); ctx.stroke();
      ctx.beginPath(); ctx.arc(x, y, Math.min(3.2, dx / 3), 0, TAU);
      if (clipped) ctx.stroke(); else ctx.fill();
    });
  }

  // The z-plane: axes, the unit circle, and a mapping between z and pixels.
  function plane(g, c, R) {
    const { ctx, w, h } = g;
    const s = Math.min(w, h) / 2 / R, cx = w / 2, cy = h / 2;
    const toPx = (re, im) => [cx + re * s, cy - im * s];
    ctx.strokeStyle = c.line;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, cy); ctx.lineTo(w, cy);
    ctx.moveTo(cx, 0); ctx.lineTo(cx, h);
    ctx.stroke();
    ctx.strokeStyle = c.soft;
    ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.arc(cx, cy, s, 0, TAU); ctx.stroke();
    ctx.setLineDash([]);
    return { toPx, fromPx: (x, y) => [(x - cx) / s, (cy - y) / s], s };
  }

  // Pole = ×, zero = ○ (the standard notation).
  function mark(ctx, [x, y], kind, color, r) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    if (kind === "pole") {
      ctx.moveTo(x - r, y - r); ctx.lineTo(x + r, y + r);
      ctx.moveTo(x - r, y + r); ctx.lineTo(x + r, y - r);
    } else {
      ctx.arc(x, y, r, 0, TAU);
    }
    ctx.stroke();
  }

  const fmt = (x, d) => (x < 0 ? "−" : "") + Math.abs(x).toFixed(d === undefined ? 2 : d);
  // "+ 1.45" / "− 0.81", for writing out equations term by term.
  const signed = (x, d) => (x < 0 ? " − " : " + ") + Math.abs(x).toFixed(d === undefined ? 2 : d);

  /* ---------- Step 1: one spiral ---------- */
  function spiral(root) {
    const N = 40, R = 1.35;
    const pl = root.querySelector(".zt-plane");
    const seq = root.querySelector(".fx-plot");
    const rIn = root.querySelector('[name="r"]');
    const tIn = root.querySelector('[name="theta"]');
    const eq = root.querySelector(".fx-eq");
    const note = root.querySelector(".fx-note");

    function draw() {
      const r = +rIn.value, th = +tIn.value * Math.PI;
      root.querySelector('[for="r"]').textContent = r.toFixed(2);
      root.querySelector('[for="theta"]').textContent = (+tIn.value).toFixed(2) + "π";
      eq.textContent = "z = " + r.toFixed(2) + " · e^(i·" + (+tIn.value).toFixed(2) + "π)   →   x[n] = " +
        r.toFixed(2) + "ⁿ · cos(" + (+tIn.value).toFixed(2) + "π · n)";

      const c = palette();
      const g = fit(pl), m = plane(g, c, R);
      // The powers z^0, z^1, z^2 … as dots, joined so you can see the spiral.
      g.ctx.strokeStyle = c.tealSoft;
      g.ctx.lineWidth = 1;
      g.ctx.beginPath();
      for (let n = 0; n < N; n++) {
        const [x, y] = m.toPx(r ** n * Math.cos(th * n), r ** n * Math.sin(th * n));
        if (n === 0) g.ctx.moveTo(x, y); else g.ctx.lineTo(x, y);
      }
      g.ctx.stroke();
      g.ctx.fillStyle = c.teal;
      for (let n = 0; n < N; n++) {
        const [x, y] = m.toPx(r ** n * Math.cos(th * n), r ** n * Math.sin(th * n));
        g.ctx.beginPath(); g.ctx.arc(x, y, 2, 0, TAU); g.ctx.fill();
      }
      const [px, py] = m.toPx(r * Math.cos(th), r * Math.sin(th));
      g.ctx.fillStyle = c.signal;
      g.ctx.beginPath(); g.ctx.arc(px, py, 5.5, 0, TAU); g.ctx.fill();

      const vals = [];
      for (let n = 0; n < N; n++) vals.push(r ** n * Math.cos(th * n));
      stems(fit(seq), c, vals, 1.2, c.signal);

      if (Math.abs(r - 1) < 0.005) {
        note.textContent = "Right on the circle: it never grows, never fades. That’s a plain sine wave — the only kind the Fourier post used.";
      } else if (r < 1) {
        note.textContent = "Inside the circle: every step shrinks it, so it dies out. The closer to the edge, the longer it lingers.";
      } else {
        note.textContent = "Outside the circle: every step grows it. It runs off the chart and never comes back.";
      }
    }
    [rIn, tIn].forEach((el) => el.addEventListener("input", draw));
    return draw;
  }

  /* ---------- Steps 2 & 3: drag poles (and zeros) ---------- */
  function poles(root) {
    const N = 40, R = 1.35, W = 240; // W = frequency-response samples
    const withZero = root.dataset.zeros === "1";
    const pl = root.querySelector(".zt-plane");
    const imp = root.querySelector(".fx-plot");
    const fr = root.querySelector(".zt-freq");
    const eq = root.querySelector(".fx-eq");
    const note = root.querySelector(".fx-note");
    const legend = root.querySelector(".zt-freq-legend");

    // Upper-half-plane positions; the mirror-image partner is implied.
    const start = {
      pole: withZero ? [0.85 * Math.cos(0.3 * Math.PI), 0.85 * Math.sin(0.3 * Math.PI)]
                     : [0.9 * Math.cos(0.2 * Math.PI), 0.9 * Math.sin(0.2 * Math.PI)],
      zero: [Math.cos(0.65 * Math.PI) * 0.7, Math.sin(0.65 * Math.PI) * 0.7],
    };
    const pts = { pole: start.pole.slice(), zero: start.zero.slice() };
    let active = "pole", map = null;

    // y[n] = b0 x[n] + b1 x[n−1] + b2 x[n−2] + a1 y[n−1] + a2 y[n−2]
    function coeffs() {
      const [pr, pi] = pts.pole, [zr, zi] = pts.zero;
      return {
        a1: 2 * pr, a2: -(pr * pr + pi * pi),
        b1: withZero ? -2 * zr : 0, b2: withZero ? zr * zr + zi * zi : 0,
      };
    }

    function draw() {
      const c = palette(), k = coeffs();
      const g = fit(pl);
      map = plane(g, c, R);
      const [pr, pi] = pts.pole, [zr, zi] = pts.zero;
      if (withZero) {
        mark(g.ctx, map.toPx(zr, zi), "zero", c.teal, 6);
        mark(g.ctx, map.toPx(zr, -zi), "zero", c.tealSoft, 6);
      }
      mark(g.ctx, map.toPx(pr, pi), "pole", c.signal, 6);
      mark(g.ctx, map.toPx(pr, -pi), "pole", c.soft, 6);
      if (document.activeElement === pl) {
        const [x, y] = map.toPx(...pts[active]);
        g.ctx.strokeStyle = c.ink; g.ctx.lineWidth = 1;
        g.ctx.strokeRect(x - 11, y - 11, 22, 22);
      }

      // Impulse response: feed in a single 1 and run the recipe.
      const x = (n) => (n === 0 ? 1 : 0), y = [];
      for (let n = 0; n < N; n++) {
        y.push(x(n) + k.b1 * x(n - 1) + k.b2 * x(n - 2) +
          k.a1 * (n >= 1 ? y[n - 1] : 0) + k.a2 * (n >= 2 ? y[n - 2] : 0));
      }
      const pr2 = Math.hypot(pr, pi);
      const stable = pr2 < 0.995;
      const yMax = 1.15 * Math.max(1, ...y.slice(0, stable ? N : 12).map(Math.abs).filter(isFinite));
      stems(fit(imp), c, y, yMax, c.signal);

      // Frequency response = |H| walking the top half of the unit circle, 0 → π.
      const f = fit(fr);
      const dist = (w, [a, b]) => Math.hypot(Math.cos(w) - a, Math.sin(w) - b) * Math.hypot(Math.cos(w) - a, Math.sin(w) + b);
      if (stable) {
        const mags = [];
        for (let i = 0; i <= W; i++) {
          const w = (i / W) * Math.PI;
          mags.push((withZero ? dist(w, pts.zero) : 1) / Math.max(1e-9, dist(w, pts.pole)));
        }
        const peak = Math.max(...mags), floor = f.h - 14;
        f.ctx.strokeStyle = c.line; f.ctx.lineWidth = 1;
        f.ctx.beginPath(); f.ctx.moveTo(0, floor); f.ctx.lineTo(f.w, floor); f.ctx.stroke();
        f.ctx.fillStyle = c.tealSoft;
        f.ctx.globalAlpha = 0.25;
        f.ctx.beginPath(); f.ctx.moveTo(0, floor);
        mags.forEach((m, i) => f.ctx.lineTo((i / W) * f.w, floor - (m / peak) * (floor - 6)));
        f.ctx.lineTo(f.w, floor); f.ctx.closePath(); f.ctx.fill();
        f.ctx.globalAlpha = 1;
        f.ctx.strokeStyle = c.teal; f.ctx.lineWidth = 2;
        f.ctx.beginPath();
        mags.forEach((m, i) => {
          const px = (i / W) * f.w, py = floor - (m / peak) * (floor - 6);
          if (i === 0) f.ctx.moveTo(px, py); else f.ctx.lineTo(px, py);
        });
        f.ctx.stroke();
        f.ctx.font = "10px " + getComputedStyle(root).fontFamily;
        f.ctx.fillStyle = c.soft;
        f.ctx.textAlign = "left"; f.ctx.fillText("slow", 0, f.h - 2);
        f.ctx.textAlign = "right"; f.ctx.fillText("fast", f.w, f.h - 2);
        legend.textContent = "how strongly it responds to each frequency, slowest → fastest the samples allow";
      } else {
        f.ctx.font = "italic 13px " + getComputedStyle(root).fontFamily;
        f.ctx.fillStyle = c.soft;
        f.ctx.textAlign = "center";
        f.ctx.fillText("unstable — it never settles, so there’s no response to draw", f.w / 2, f.h / 2);
        legend.textContent = "";
      }

      eq.textContent = "y[n] = x[n]" +
        (withZero ? signed(k.b1) + "·x[n−1]" + signed(k.b2) + "·x[n−2]" : "") +
        signed(k.a1) + "·y[n−1]" + signed(k.a2) + "·y[n−2]";

      const ang = Math.atan2(pi, pr) / Math.PI;
      if (!stable) {
        note.textContent = "The pole is on or outside the circle — the feedback feeds itself and the output blows up. Every real filter keeps its poles inside.";
      } else if (withZero && Math.abs(Math.hypot(zr, zi) - 1) < 0.002) {
        note.textContent = "Zero on the circle: that frequency is gone completely — the response touches the floor at exactly that angle.";
      } else if (withZero) {
        note.textContent = "Poles push the response up, zeros pull it down. Try dropping the zero right onto the dashed circle.";
      } else if (pr2 > 0.95) {
        note.textContent = "Close to the circle: it rings for a long time, and the response has a sharp peak. Same thing, seen two ways.";
      } else if (ang < 0.03) {
        note.textContent = "On the real axis there’s no angle, so no oscillation — it just decays (or, past the circle, grows).";
      } else {
        note.textContent = "The pole’s angle sets the pitch it rings at; its distance from the edge sets how long it rings.";
      }
    }

    // Dragging: grab whichever mark (or its mirror) is nearest the pointer.
    function zAt(e) {
      const r = pl.getBoundingClientRect();
      return map.fromPx(e.clientX - r.left, e.clientY - r.top);
    }
    function place(name, re, im) {
      im = Math.abs(im);
      const m = Math.hypot(re, im);
      if (m > 1.25) { re *= 1.25 / m; im *= 1.25 / m; }
      // Zeros snap onto the unit circle when close — that's where they're most useful.
      if (name === "zero" && Math.abs(Math.hypot(re, im) - 1) < 0.035) {
        const a = Math.atan2(im, re); re = Math.cos(a); im = Math.sin(a);
      }
      pts[name] = [re, im];
    }
    let dragging = null;
    pl.addEventListener("pointerdown", (e) => {
      e.preventDefault();
      const [re, im] = zAt(e);
      const names = withZero ? ["pole", "zero"] : ["pole"];
      let best = names[0], bd = Infinity;
      names.forEach((n) => {
        const [a, b] = pts[n], d = Math.min(Math.hypot(re - a, im - b), Math.hypot(re - a, im + b));
        if (d < bd) { bd = d; best = n; }
      });
      dragging = active = best;
      try { pl.setPointerCapture(e.pointerId); } catch (_) { /* capture is a nicety */ }
      place(best, re, im); draw();
    });
    pl.addEventListener("pointermove", (e) => {
      if (!dragging) return;
      const [re, im] = zAt(e);
      place(dragging, re, im); draw();
    });
    const stop = () => { dragging = null; };
    pl.addEventListener("pointerup", stop);
    pl.addEventListener("pointercancel", stop);

    // Keyboard: arrows nudge the active mark (shift = bigger steps); space, "p" or "z" switches it.
    pl.addEventListener("keydown", (e) => {
      const step = e.shiftKey ? 0.1 : 0.02, [re, im] = pts[active];
      const moves = { ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, step], ArrowDown: [0, -step] };
      if (moves[e.key]) {
        e.preventDefault();
        place(active, re + moves[e.key][0], im + moves[e.key][1]); draw();
      } else if (withZero && (e.key === " " || e.key === "p" || e.key === "z")) {
        e.preventDefault();
        active = e.key === "p" ? "pole" : e.key === "z" ? "zero" : active === "pole" ? "zero" : "pole";
        draw();
      }
    });
    pl.addEventListener("focus", () => draw());
    pl.addEventListener("blur", () => draw());

    root.querySelector('[data-act="reset"]').addEventListener("click", () => {
      pts.pole = start.pole.slice(); pts.zero = start.zero.slice(); draw();
    });
    return draw;
  }

  /* ---------- Step 4: a live 60 Hz notch ---------- */
  function notch(root) {
    const FS = 5000, N = 5000;           // 1 s at 5 kHz
    const SHOW = 2500;                   // plot the first 0.5 s
    const SPEC_N = 1000, BINS = 200;     // spectrum of the last 0.2 s: 5 Hz bins, 0–1000 Hz
    const trace = root.querySelector(".fx-plot");
    const spec = root.querySelector(".fx-scope-spec");
    const sIn = root.querySelector('[name="sharp"]');
    const sOut = root.querySelector('[for="sharp"]');
    const harm = root.querySelector('[name="harm"]');
    const rmsOut = root.querySelector(".fx-rms");
    const eq = root.querySelector(".fx-eq");
    const note = root.querySelector(".fx-note");

    // Same recipe as the Fourier post's powered-off trace: mains hum,
    // a few harmonics, and broadband noise. Seeded so every reader sees the same thing.
    let seed = 7;
    const rand = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
    const gauss = () => Math.sqrt(-2 * Math.log(rand() + 1e-12)) * Math.cos(TAU * rand());
    const hum = [[60, 1.0, 0.3], [120, 0.06, 1.1], [180, 0.35, 2.0], [300, 0.18, 0.7], [420, 0.1, 4.0], [540, 0.05, 1.5]];
    const raw = new Float32Array(N), humOnly = new Float32Array(N);
    for (let j = 0; j < N; j++) {
      const t = j / FS;
      let y = 0;
      hum.forEach(([f, a, p]) => { y += a * Math.sin(TAU * f * t + p); });
      raw[j] = y + 0.55 * gauss();
    }
    const out = new Float32Array(N), humOut = new Float32Array(N);

    // One notch = zeros on the circle at ±ω0, poles at radius r just inside them.
    const section = (f, r) => {
      const cw = Math.cos((TAU * f) / FS);
      return { b0: 1, b1: -2 * cw, b2: 1, a1: 2 * r * cw, a2: -r * r };
    };
    // Run the recipe one sample at a time, exactly as a microcontroller would.
    function run(secs, src, dst) {
      dst.set(src);
      secs.forEach((s) => {
        let x1 = 0, x2 = 0, y1 = 0, y2 = 0;
        for (let j = 0; j < N; j++) {
          const x0 = dst[j];
          const y0 = s.b0 * x0 + s.b1 * x1 + s.b2 * x2 + s.a1 * y1 + s.a2 * y2;
          x2 = x1; x1 = x0; y2 = y1; y1 = y0;
          dst[j] = y0;
        }
      });
    }
    function dft(sig) {
      const mag = new Float32Array(BINS + 1), o = N - SPEC_N;
      for (let k = 1; k <= BINS; k++) {
        let r = 0, m = 0;
        for (let j = 0; j < SPEC_N; j++) {
          const a = (TAU * k * j) / SPEC_N;
          r += sig[o + j] * Math.cos(a); m -= sig[o + j] * Math.sin(a);
        }
        mag[k] = (2 * Math.hypot(r, m)) / SPEC_N;
      }
      return mag;
    }
    const rawMag = dft(raw);
    const rms = (sig, a, b) => { let s = 0; for (let j = a; j < b; j++) s += sig[j] ** 2; return Math.sqrt(s / (b - a)); };
    const rms0 = rms(raw, N - SPEC_N, N);
    const humBins = hum.map(([f]) => Math.round((f * SPEC_N) / FS));

    function draw() {
      const r = 1 - 10 ** -(+sIn.value);
      const freqs = harm.checked ? [60, 180, 300, 420, 540] : [60];
      const secs = freqs.map((f) => section(f, r));
      run(secs, raw, out);
      // The filter is linear, so running it on just the hum it's aimed at
      // shows exactly how much of that hum is still getting through.
      for (let j = 0; j < N; j++) {
        let y = 0;
        hum.forEach(([f, a, p]) => { if (freqs.includes(f)) y += a * Math.sin((TAU * f * j) / FS + p); });
        humOnly[j] = y;
      }
      run(secs, humOnly, humOut);
      const bw = ((1 - r) * FS) / Math.PI;            // notch width, Hz (−3 dB)
      const settle = Math.log(0.01) / Math.log(r) / FS; // seconds for the ringing to fall to 1%
      sOut.textContent = r.toFixed(r > 0.995 ? 4 : 3);

      const c = palette();
      const g = fit(trace);
      axis(g, c);
      const line = (sig, color, width) => {
        g.ctx.strokeStyle = color; g.ctx.lineWidth = width; g.ctx.lineJoin = "round";
        g.ctx.beginPath();
        for (let j = 0; j < SHOW; j++) {
          const x = (j / (SHOW - 1)) * g.w, y = g.h / 2 - (sig[j] / 3.2) * (g.h / 2 - 6);
          if (j === 0) g.ctx.moveTo(x, y); else g.ctx.lineTo(x, y);
        }
        g.ctx.stroke();
      };
      line(raw, c.line, 1);
      line(humOut, c.signal, 2);

      // Spectrum after filtering (bars) with the filter's own response drawn over it.
      const mag = dft(out), s = fit(spec), bwPx = s.w / BINS, top = 16, floor = s.h - 14;
      const peak = Math.max(...rawMag);
      for (let k = 1; k <= BINS; k++) {
        const x = (k - 1) * bwPx, wPx = Math.max(1, bwPx - 1);
        s.ctx.fillStyle = c.line;
        const rh = (rawMag[k] / peak) * (floor - top);
        s.ctx.fillRect(x, floor - rh, wPx, rh);
        const bh = (mag[k] / peak) * (floor - top);
        s.ctx.fillStyle = humBins.includes(k) ? c.signal : c.tealSoft;
        s.ctx.fillRect(x, floor - bh, wPx, bh);
      }
      // |H(e^{iω})| across 0–1000 Hz, top of the chart = gain of 1.
      s.ctx.strokeStyle = c.teal; s.ctx.lineWidth = 1.5;
      s.ctx.beginPath();
      for (let i = 0; i <= s.w; i++) {
        const w = (TAU * (i / s.w) * 1000) / FS;
        let H = 1;
        secs.forEach((q) => {
          const cr = Math.cos(w), sr = Math.sin(w), c2 = Math.cos(2 * w), s2 = Math.sin(2 * w);
          const nr = q.b0 + q.b1 * cr + q.b2 * c2, ni = -(q.b1 * sr + q.b2 * s2);
          const dr = 1 - q.a1 * cr - q.a2 * c2, di = q.a1 * sr + q.a2 * s2;
          H *= Math.hypot(nr, ni) / Math.hypot(dr, di);
        });
        const y = 4 + (1 - Math.min(1.05, H)) * (floor - 4);
        if (i === 0) s.ctx.moveTo(i, y); else s.ctx.lineTo(i, y);
      }
      s.ctx.stroke();
      s.ctx.font = "10px " + getComputedStyle(root).fontFamily;
      s.ctx.fillStyle = c.soft;
      [0, 250, 500, 750, 1000].forEach((f) => {
        s.ctx.textAlign = f === 0 ? "left" : f === 1000 ? "right" : "center";
        s.ctx.fillText(f, (f / 1000) * s.w, s.h - 2);
      });

      const after = rms(out, N - SPEC_N, N);
      rmsOut.textContent = "noise " + rms0.toFixed(2) + " → " + after.toFixed(2) + " mV rms (once settled)";
      const q = secs[0];
      eq.textContent = "y[n] = x[n]" + signed(q.b1, 3) + "·x[n−1]" + signed(q.b2, 3) +
        "·x[n−2]" + signed(q.a1, 3) + "·y[n−1]" + signed(q.a2, 3) + "·y[n−2]";

      const ms = Math.round(settle * 1000), hz = bw < 10 ? bw.toFixed(1) : Math.round(bw);
      if (bw > 60) {
        note.textContent = "Notch is about " + hz + " Hz wide — it’s chewing into everything near 60 Hz, not just the hum. Settles in " + ms + " ms, though.";
      } else if (settle > 0.25) {
        note.textContent = "Surgical: about " + hz + " Hz wide. But watch the bright line — the hum takes " + ms +
          " ms to fade. That’s the pole sitting a hair inside the circle, ringing exactly like step 2.";
      } else {
        note.textContent = "About " + hz + " Hz wide, settles in " + ms + " ms. Narrower costs you settling time; faster costs you a wider bite.";
      }
    }
    sIn.addEventListener("input", draw);
    harm.addEventListener("change", draw);
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

  const kinds = { spiral, poles, notch };
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
