---
layout: post
title: "Every wave is just a pile of sine waves. Try it."
---

<link rel="stylesheet" href="/assets/css/fourier.css">

Fourier decomposition is one of those ideas I understood on paper for years before I actually *got* it, and what finally did it wasn't another derivation — it was messing around until the answer felt obvious. So instead of explaining it, this post hands you the knobs. Three steps, no math required, though the math is sitting right underneath if you want it.

## 1. Meet one sine wave

Everything here is built out of one ingredient. A sine wave has three dials: how tall it is (amplitude), how many times it repeats in the window (frequency), and where in its cycle it starts (phase). Play with them until they feel boring. That's the point — they should be boring.

<div class="fx" data-fx="one">
  <p class="eyebrow">Step 1 · One wave</p>
  <canvas class="fx-plot" role="img" aria-label="A single sine wave controlled by the sliders below"></canvas>
  <p class="fx-eq"></p>
  <label class="fx-row"><span class="fx-k">Amplitude</span><input type="range" name="amp" min="0" max="1" step="0.01" value="0.8"><output for="amp"></output></label>
  <label class="fx-row"><span class="fx-k">Frequency</span><input type="range" name="freq" min="1" max="8" step="1" value="1"><output for="freq"></output></label>
  <label class="fx-row"><span class="fx-k">Phase</span><input type="range" name="phase" min="0" max="6.2832" step="0.01" value="0"><output for="phase"></output></label>
</div>

## 2. Now build a square wave out of them

Here's the claim that sounds wrong the first time you hear it: add up enough smooth, round sine waves and you can make a square wave — flat tops, sharp corners and all. You get eight sine waves to work with. Wave 1× repeats once across the window, 2× repeats twice, and so on. The dashed line is the target. See how close you can get.

<div class="fx" data-fx="square">
  <p class="eyebrow">Step 2 · Match the square</p>
  <canvas class="fx-plot" role="img" aria-label="Your sum of sine waves drawn over a dashed square-wave target"></canvas>
  <div class="fx-bar">
    <span class="fx-meter" aria-hidden="true"><span></span></span>
    <span class="fx-score" aria-live="polite"></span>
  </div>
  <div class="fx-harmonics"></div>
  <p class="fx-note" aria-live="polite"></p>
  <div class="fx-bar">
    <label><input type="checkbox" name="parts"> show the individual waves</label>
    <button type="button" class="btn" data-act="reset">Reset</button>
    <button type="button" class="btn" data-act="solve">Show me</button>
  </div>
</div>

If you got close, you probably found the pattern on your own: only the odd ones matter, and each one gets smaller — 1, ⅓, ⅕, ⅐ of the first. That recipe *is* the square wave. Fourier's actual insight is that every repeating shape has a recipe like that, and there's exactly one right answer for it.

## 3. Draw anything

You don't have to find the recipe by hand, though. The Fourier transform reads it straight off the shape. Draw whatever you want in the box below (or pick a preset), then slide the number of waves up from 1. The bars underneath are the recipe — how much of each wave the shape needs. Lit bars are the ones currently in use.

<div class="fx" data-fx="draw">
  <p class="eyebrow">Step 3 · Draw, then decompose</p>
  <canvas class="fx-pad" role="img" aria-label="Drawing area: drag to draw a shape; the rebuilt shape is overlaid"></canvas>
  <canvas class="fx-spec" aria-hidden="true"></canvas>
  <p class="fx-legend">dashed = what you drew · <b>solid</b> = rebuilt from sine waves · bars = how much of each wave, 1× to 64×</p>
  <label class="fx-row"><span class="fx-k">Waves</span><input type="range" name="terms" min="1" max="64" step="1" value="3"><output for="terms"></output></label>
  <div class="fx-bar">
    <span class="fx-score" aria-live="polite"></span>
  </div>
  <div class="fx-bar">
    <button type="button" class="btn" data-preset="heartbeat">Heartbeat</button>
    <button type="button" class="btn" data-preset="square">Square</button>
    <button type="button" class="btn" data-preset="saw">Sawtooth</button>
    <button type="button" class="btn" data-preset="triangle">Triangle</button>
  </div>
</div>

A few things worth noticing while you play: smooth shapes (try the triangle) are basically done after a handful of waves, while anything with a sharp jump needs a long tail of them. And no matter how many you add to the square, the rebuild overshoots a little at every corner — that's the Gibbs phenomenon, and it never fully goes away.

That last observation is also why this matters outside a math class. Most of the "shape" of a real signal usually lives in a few big bars, and the long tail of tiny ones is mostly noise. Keep the big ones, drop the rest, and you've just compressed an audio file, cleaned up a sensor reading, or — same idea, different basis — done the [SVD signal extraction](/2017/10/09/SVD_Introduction.html) from my first post here.

<script src="/assets/js/fourier.js" defer></script>
