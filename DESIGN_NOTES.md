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
  post is about the glow, so the treatment gets out of its way there.
- **Layouts**: `_layouts/home.html` (hero + content, used by `index.md`),
  `_layouts/page.html` (generic — About, Research, Notes index),
  `_layouts/post.html`. All extend `_layouts/default.html`, which owns the
  nav/footer/`<head>`/asset includes and the nav's active-link logic.
- **Notes list**: home shows a compact `.notes-list`; `/blog` shows fuller
  `.note-cards` with a one-line excerpt pulled from each post's Jekyll-generated
  `excerpt` (no hand-duplicated summary text to keep in sync).

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
