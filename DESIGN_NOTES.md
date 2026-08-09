# Visual refresh — deferred, not forgotten

Content/structure got a rebuild in Aug 2026 (see git log). The visual layer
is untouched — still `jekyll-theme-leap-day`, a ~decade-old default GitHub
Pages theme. Notes for whenever that gets picked up:

- **Theme**: `jekyll-theme-leap-day` (set in `_config.yml`) is dated and its
  SCSS already throws Dart Sass `@import` deprecation warnings on build.
  Either pick a more modern minimal Jekyll theme, or replace it with a small
  custom stylesheet — the site is only 4 pages, doesn't need much.
- **Mobile**: never explicitly verified. The theme predates mobile-first
  conventions; check readability and nav behavior on a phone-width viewport
  before/after any theme swap.
- **Contact section**: currently plain LinkedIn link + a JS-rendered email
  on the homepage. Fine functionally, but hasn't had any visual attention.
- **Homepage portrait / photos**: `assets/images/Ted2.jpg` and
  `Ted_Hiking.jpg` are old; swap for current photos whenever convenient.
- **Notes list styling**: post list is currently a bare `<ul>` on the
  homepage and `/blog`. Once there are a few real posts, worth giving them
  slightly more visual weight (date styling, maybe a one-line excerpt).

None of this is urgent — the priority was fixing what the site *says*, not
what it looks like. Pick this up whenever there's appetite for a design pass.
