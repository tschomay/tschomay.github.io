(function () {
  "use strict";

  // Homepage hero: draws a few noisy curves that resolve into one signal line.
  var canvas = document.getElementById("wave");
  if (canvas) {
    var ctx = canvas.getContext("2d");
    var reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    var dpr = Math.min(window.devicePixelRatio || 1, 2);

    function cssVar(name) {
      return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    }

    function size() {
      var rect = canvas.parentElement.getBoundingClientRect();
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      canvas.style.width = rect.width + "px";
      canvas.style.height = rect.height + "px";
    }
    size();
    window.addEventListener("resize", size);

    var noise = [
      { amp: 0.16, freq: 1.4, phase: 0.3, speed: 0.00034 },
      { amp: 0.11, freq: 2.3, phase: 2.1, speed: -0.00041 },
      { amp: 0.20, freq: 0.8, phase: 4.0, speed: 0.00026 },
      { amp: 0.09, freq: 3.1, phase: 1.2, speed: -0.00052 }
    ];
    var signalWave = { amp: 0.14, freq: 1.0 };

    function ease(t) { return 1 - Math.pow(1 - t, 3); }

    var start = null;
    var DURATION = 1800;

    function draw(ts) {
      if (start === null) start = ts;
      var elapsed = reduce ? DURATION : ts - start;
      var p = Math.min(1, elapsed / DURATION);
      var e = ease(p);

      var w = canvas.width, h = canvas.height, mid = h * 0.58;
      ctx.clearRect(0, 0, w, h);

      var inkSoft = cssVar("--ink-soft");
      var teal = cssVar("--teal");
      var signalColor = cssVar("--signal");

      noise.forEach(function (n, i) {
        var op = 0.5 - e * 0.4;
        var ampNow = n.amp * (1 - e * 0.55);
        ctx.beginPath();
        for (var x = 0; x <= w; x += 6) {
          var t = x / w;
          var y = mid + Math.sin(t * Math.PI * 2 * n.freq + n.phase + elapsed * n.speed) * ampNow * h;
          if (x === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.strokeStyle = i % 2 === 0 ? inkSoft : teal;
        ctx.globalAlpha = op;
        ctx.lineWidth = 1 * dpr;
        ctx.stroke();
      });

      ctx.beginPath();
      var driftSpeed = 0.00012;
      for (var x = 0; x <= w; x += 4) {
        var t = x / w;
        var flatten = 1 - e;
        var y = mid
          + Math.sin(t * Math.PI * 2 * signalWave.freq + elapsed * driftSpeed) * signalWave.amp * h * e
          + Math.sin(t * Math.PI * 6 + elapsed * 0.0009) * signalWave.amp * h * 0.4 * flatten;
        if (x === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.strokeStyle = signalColor;
      ctx.globalAlpha = 0.15 + e * 0.85;
      ctx.lineWidth = 2.4 * dpr;
      ctx.stroke();
      ctx.globalAlpha = 1;

      if (reduce) return;
      requestAnimationFrame(draw);
    }
    requestAnimationFrame(draw);
  }
})();
