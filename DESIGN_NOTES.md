# Visual system — "Signal & Noise" (Aug 2026)

Content got a rebuild in Aug 2026, and the visual layer followed right after with a
custom design built from the site's own material rather than a generic theme swap.
Ted's 2017 post is literally titled "Demonstration of Signal Extraction with SVD" —
a method for pulling one clean pattern out of noisy data. That's also the site's
core thesis (what you do vs. how you do it). The whole visual system is built from
that one idea. It was proposed as an Artifact mockup and approved by Ted before any
code was touched — if you need to see the original pitch, ask Ted for the link.

## What's implemented

- Dropped `jekyll-theme-leap-day` entirely (removed the `theme:` key in
  `_config.yml`). The site now ships its own plain CSS at `assets/css/style.css`
  and vanilla JS at `assets/js/site.js` — no jQuery, no Sass, no external theme
  gem. This also resolves the old Dart Sass `@import` deprecation warnings.
- **Palette**: ink/paper/surface neutrals plus a brass "signal" accent and a deep
  teal secondary, defined as CSS custom properties on `:root` with a
  `prefers-color-scheme: dark` override. The `--signal` shade is decorative-only
  (borders, underlines, the canvas wave); `--signal-text` is a darker variant used
  anywhere the color sits on actual text, chosen to clear WCAG AA contrast on the
  light background.
- **Two typefaces, deliberately split**: a tight technical system-sans for
  headlines/labels/eyebrows (the "what") and a warm serif for body paragraphs (the
  "how") — see `--font-display` / `--font-body` in `style.css`. System font stacks
  only, no webfonts to keep the site dependency-free.
- **One shape, reused**: a notched top-left corner (`.notch`, a `clip-path`
  utility) on portrait frames, post images, and the primary button — instead of
  rounded corners everywhere.
- **Homepage hero animation** (`assets/js/site.js`, canvas `#wave`): draws several
  noisy curves that resolve into a single signal line on load. Skips straight to
  the resolved state under `prefers-reduced-motion: reduce`.
- **Portraits get a duotone treatment**: an SVG filter (`#duotone`, defined once in
  `_layouts/default.html`, referenced via `style="filter:url(#duotone)"`) maps
  shadows to ink and highlights to the brass signal color. Applied to `Ted2.jpg`
  (home hero) and `Ted_Hiking.jpg` (about) to unify the older photos into the
  palette. Notes post photos (Auralux) are deliberately left full color — that
  post is about the glow, so the treatment gets out of its way there. See "How
  the duotone treatment works" below before adding it to a new photo.
- **Layouts**: `_layouts/home.html` (hero + content, used by `index.md`),
  `_layouts/page.html` (generic — About, Research, Notes index),
  `_layouts/post.html`. All extend `_layouts/default.html`, which owns the
  nav/footer/`<head>`/asset includes and the nav's active-link logic.
- **Notes list**: home shows a compact `.notes-list`; `/blog` shows fuller
  `.note-cards` with a one-line excerpt pulled from each post's Jekyll-generated
  `excerpt` (no hand-duplicated summary text to keep in sync).

## How the duotone treatment works

Important: this is a **live CSS/SVG filter applied in the browser**, not a
pre-edited image file. There is no image-processing step and no separate
"duotone version" of any photo saved to disk — the original JPG/PNG stays
untouched in `assets/images/`, and the browser recolors it on the fly. Adding
the treatment to a new photo is a one-line HTML change, not an image edit.

The filter is defined once, near the top of `_layouts/default.html`, as a
hidden inline `<svg>` (so it doesn't need a request to a separate file, and
`filter:url(#duotone)` can reference it from any page):

```html
<svg width="0" height="0" style="position:absolute" aria-hidden="true">
  <filter id="duotone" color-interpolation-filters="sRGB">
    <feColorMatrix type="matrix" values="0.299 0.587 0.114 0 0  0.299 0.587 0.114 0 0  0.299 0.587 0.114 0 0  0 0 0 1 0"/>
    <feComponentTransfer>
      <feFuncR type="table" tableValues="0.086 0.851"/>
      <feFuncG type="table" tableValues="0.129 0.573"/>
      <feFuncB type="table" tableValues="0.114 0.255"/>
    </feComponentTransfer>
  </filter>
</svg>
```

Two steps happen:

1. **`feColorMatrix`** flattens the photo to grayscale using standard luminance
   weights (0.299R + 0.587G + 0.114B). This is generic — it never needs to
   change regardless of what colors you're mapping to.
2. **`feComponentTransfer`** with `type="table"` remaps that grayscale value
   per channel. Each `tableValues="A B"` linearly interpolates: black pixels
   (0.0) become `A`, white pixels (1.0) become `B`, everything in between
   blends. So `feFuncR`/`feFuncG`/`feFuncB` together define **two colors**:
   the shadow color (all three `A` values) and the highlight color (all three
   `B` values). Here that's ink `#16211D` for shadows → brass `#D9A441` for
   highlights (a slightly warmer, more saturated brass than the `--signal`
   CSS token, chosen because a photo highlight wants more pop than
   text-safe accent color does).

**To apply it to a new photo**, just add the filter (and, to match the rest of
the site's portrait framing, the `.notch` class) to the `<img>` tag:

```html
<img src="/assets/images/whatever.jpg" alt="..." class="notch" style="filter:url(#duotone);">
```

No new markup, no new filter definition needed — reuse the existing
`#duotone` filter for any personal/portrait photo. Leave project or
screenshot images (like the Auralux photos in Notes posts) untouched — the
full-color-vs-duotone contrast between "Ted" photos and "project" photos is a
deliberate rule (see above), not an oversight.

**If the ink/brass hex values ever change** (e.g. the palette tokens in
`style.css` get revised), the `tableValues` above will need recomputing by
hand — SVG filter primitives can't read CSS custom properties, so nothing
here updates automatically when `:root` does. The recipe, given any shadow
hex and highlight hex:

1. Split each hex into R, G, B (0–255).
2. Divide each channel by 255 to get a 0–1 fraction (three decimals is
   plenty of precision).
3. `feFuncR tableValues="<shadow R> <highlight R>"`, and the same pattern for
   G and B.

Worked example for the values above: ink `#16211D` = rgb(22, 33, 29) →
0.086, 0.129, 0.114. Brass `#D9A441` = rgb(217, 146, 65) → 0.851, 0.573,
0.255. `color-interpolation-filters="sRGB"` on the `<filter>` is what makes
this direct hex-to-fraction math valid — without it, browsers default to
linearRGB and the numbers would need gamma correction first.

## Still open

- **Photo currency**: `Ted2.jpg` / `Ted_Hiking.jpg` are still the same older
  photos flagged before this pass — the duotone treatment helps them sit inside
  the new palette in the meantime, but swap them for current ones whenever Ted
  has new ones. Update the `portrait:` front-matter field in `index.md` and the
  `<img src>` in `about.md`.
- Nothing else from the original punch list is outstanding — theme swap, mobile
  check, and notes-list styling are all done.

## Building locally

`bundle install` fails in this repo in some sandboxed environments — an old
pinned `bundler` (1.15.4) doesn't work with Ruby 3.3 (`undefined method
'untaint'`). Fallback that works:

```
gem install jekyll jekyll-seo-tag --no-document
JEKYLL_NO_BUNDLER_REQUIRE=1 jekyll build --trace
```

`JEKYLL_NO_BUNDLER_REQUIRE=1` skips Jekyll's attempt to shell out to Bundler
(which is what actually trips the broken bundler) while still building with the
plugins installed directly as gems. The site only needs `jekyll` and
`jekyll-seo-tag` (for the `{% seo %}` tag in `_layouts/default.html`) — nothing
else in the dependency chain requires the theme gem anymore.
