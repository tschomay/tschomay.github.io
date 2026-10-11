---
layout: post
title: "Sine waves are allowed to fade out. That's the Z-transform."
---

The [Fourier post](/2026/10/10/fourier-by-hand.html) built everything out of sine waves that go on forever at the same height. Real signals don't do that. A plucked string fades, a filter rings and settles, a feedback loop with too much gain runs away. The Z-transform is what you get when you let the sine waves grow or die out too — and it turns out to be the tool for designing anything that processes a signal one sample at a time. Four steps again, same deal: play first, explanation after.

<link rel="stylesheet" href="/assets/css/fourier.css">
<link rel="stylesheet" href="/assets/css/ztransform.css">

## 1. Meet one spiral

The ingredient this time is a single complex number, *z*, and what happens when you keep multiplying by it: *z*, *z*², *z*³, and so on. Its angle says how far around the circle each step turns. Its distance from the center says whether each step shrinks things or grows them. On the left is where those powers land; on the right is the sequence you'd actually measure — the shadow they cast on the horizontal axis.

<div class="fx" data-fx="spiral">
  <p class="eyebrow">Step 1 · One spiral</p>
  <div class="zt-split">
    <canvas class="zt-plane" role="img" aria-label="The complex plane with the unit circle dashed, showing the powers of z spiraling around the origin"></canvas>
    <div class="zt-stack">
      <p class="zt-cap">the sequence x[n], n = 0 to 39</p>
      <canvas class="fx-plot" role="img" aria-label="The sequence of samples produced by z to the n"></canvas>
    </div>
  </div>
  <p class="fx-eq"></p>
  <label class="fx-row"><span class="fx-k">Radius</span><input type="range" name="r" min="0.6" max="1.12" step="0.01" value="0.9"><output for="r"></output></label>
  <label class="fx-row"><span class="fx-k">Angle</span><input type="range" name="theta" min="0" max="1" step="0.01" value="0.15"><output for="theta"></output></label>
  <p class="fx-note" aria-live="polite"></p>
  <p class="fx-legend">tip: double-tap any slider to put it back · hollow dots ran off the chart</p>
</div>

The dashed circle is the whole story. Inside it, the sequence dies out. Outside, it blows up. Exactly on it, you get a sine wave that never changes height — the only ingredient Fourier allowed. So the Fourier transform lives on that one circle, and the Z-transform is the same idea with the rest of the plane opened up.

## 2. Drag a pole

Here's why anyone cares. The simplest useful digital filter is a recipe that mixes each new input sample with the last couple of *outputs*: y[n] = x[n] + a·y[n−1] + b·y[n−2]. Feeding the output back in is what lets a filter ring, resonate, or smooth — and it's also what lets it run away. Two numbers, and it's surprisingly hard to guess what any given pair will do.

The Z-transform's trick is that the recipe's behavior is set by one point on the plane: its *pole*, the spiral from step 1 that the feedback keeps reinforcing. (It comes with a mirror-image partner below the axis, so the output stays real.) Grab the × and move it. The top-right shows what the filter does with a single click of input; the bottom shows how strongly it responds to each frequency.

<div class="fx" data-fx="poles">
  <p class="eyebrow">Step 2 · Drag the pole</p>
  <div class="zt-split">
    <canvas class="zt-plane" tabindex="0" role="img" aria-label="The z-plane with a draggable pole pair. Arrow keys move the pole."></canvas>
    <div class="zt-stack">
      <p class="zt-cap">output after a single click of input</p>
      <canvas class="fx-plot" role="img" aria-label="Impulse response of the filter"></canvas>
      <canvas class="zt-freq" role="img" aria-label="Frequency response of the filter"></canvas>
    </div>
  </div>
  <p class="fx-legend zt-freq-legend"></p>
  <p class="fx-eq"></p>
  <p class="fx-note" aria-live="polite"></p>
  <div class="fx-bar">
    <button type="button" class="btn" data-act="reset">Reset</button>
  </div>
</div>

The click response is just the spiral from step 1 again — whatever spiral you parked the pole on. And the frequency response is that same pole seen from the edge of the circle: walk around the dashed line from the right side to the left, and the response is high wherever you pass close to the pole. A pole near the circle means a sharp peak *and* a long ring, because they're the same fact. Push it past the circle and the filter is unstable, no matter what the two numbers looked like.

## 3. Add a zero

Poles hold the response up; *zeros* (the ○) pin it down. A zero is a frequency the filter refuses to pass — it comes from mixing in the last couple of *inputs* instead of outputs. Try parking it right on the dashed circle and watch what happens to the response at that angle.

<div class="fx" data-fx="poles" data-zeros="1">
  <p class="eyebrow">Step 3 · Poles and zeros</p>
  <div class="zt-split">
    <canvas class="zt-plane" tabindex="0" role="img" aria-label="The z-plane with a draggable pole pair and zero pair. Arrow keys move the selected one; space switches between them."></canvas>
    <div class="zt-stack">
      <p class="zt-cap">output after a single click of input</p>
      <canvas class="fx-plot" role="img" aria-label="Impulse response of the filter"></canvas>
      <canvas class="zt-freq" role="img" aria-label="Frequency response of the filter"></canvas>
    </div>
  </div>
  <p class="fx-legend zt-freq-legend"></p>
  <p class="fx-eq"></p>
  <p class="fx-note" aria-live="polite"></p>
  <div class="fx-bar">
    <button type="button" class="btn" data-act="reset">Reset</button>
  </div>
</div>

That's more or less all of filter design: put zeros where you want silence, poles where you want emphasis, keep the poles inside the circle. The math underneath is one move — the Z-transform turns "the previous sample" into "multiply by *z*⁻¹", so a recipe that reaches back in time becomes a fraction of polynomials in *z*. Poles are where the bottom of that fraction hits zero, zeros are where the top does, and walking around the unit circle (*z* = *e*^(*iω*)) hands you back the Fourier transform. If you want to dig in, that's the thread to pull, along with the *region of convergence* — the formal version of "inside the circle dies out."

## 4. Kill the hum, live

Back to the scope trace from the Fourier post: the setup's powered off and there's 60 Hz line noise all over it. Last time the fix was to take the whole recording, run Fourier, delete the bad bars, and put it back together. That works on a recording. It doesn't work if you need clean data *as it comes in* — say, on a microcontroller that sees one sample at a time and has no idea what's coming next.

So build a notch the step-3 way: zeros right on the circle at 60 Hz, poles just inside them at the same angle. The slider moves the poles toward the circle.

<div class="fx" data-fx="notch">
  <p class="eyebrow">Step 4 · A 60 Hz notch, one sample at a time</p>
  <canvas class="fx-plot" role="img" aria-label="The noisy scope trace, with the hum still getting through the filter drawn over it"></canvas>
  <p class="fx-legend">first 0.5 s after the filter switches on · faint = raw trace, <b>bright</b> = targeted hum still getting through · <b class="fx-rms"></b></p>
  <canvas class="fx-scope-spec" role="img" aria-label="Spectrum of the trace after filtering, with the filter's response drawn over it"></canvas>
  <p class="fx-legend">frequency, 0–1000 Hz · faint bars = before, bright bars = after · line = the filter’s response</p>
  <label class="fx-row"><span class="fx-k">Pole r</span><input type="range" name="sharp" min="1" max="3" step="0.01" value="2" aria-label="How close the poles sit to the unit circle"><output for="sharp"></output></label>
  <div class="fx-bar">
    <label><input type="checkbox" name="harm"> also notch 180, 300, 420, 540 Hz</label>
  </div>
  <p class="fx-note" aria-live="polite"></p>
  <p class="fx-eq fx-eq-foot"></p>
</div>

Pull the poles in toward the circle and the notch gets narrower — it takes out 60 Hz and leaves everything around it alone. But watch the bright line: with the poles a hair from the edge, the hum hangs on for most of a second before the filter catches up, for exactly the reason the pole in step 2 did. Back the poles off and it settles fast, but the notch gets wide enough to eat signal you wanted. There isn't a setting that wins both; you pick based on whether you care more about the first second or about what lives near 60 Hz.

And the equation under the chart is the whole filter: five numbers and two remembered samples, cheap enough to run on about anything. You'd never land on those numbers by fiddling with the recipe directly. You get them by deciding where the poles and zeros go — that's the thing the Z-transform buys you.

<script src="/assets/js/ztransform.js" defer></script>
