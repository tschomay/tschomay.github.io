# Working on this site

This is Ted Schomay's personal site. It got a full rebuild in Aug 2026 after
sitting mostly untouched since 2022 (see git log). Read this before editing
copy, adding a Notes post, or restructuring anything — it captures decisions
that were deliberate, not default, so they don't get silently undone by a
future edit.

## What this site is actually for

Ted isn't job hunting. The site isn't a resume mirror or a "digital business
card" — that's just LinkedIn with extra steps, and LinkedIn already does
static facts better. What a personal site can do that LinkedIn can't is show
a personal touch: how Ted actually thinks, what he's like to build things
with, and the judgment behind the work — not just the résumé line.

The thesis, stated on the homepage and meant to run through everything here:

> What you do is only part of the picture — it's how you do it that has the
> bigger impact.

Concretely, that means the site is optimized to demonstrate connecting
technical depth to business judgment, and personality/working style — not to
list credentials. The homepage's "What I'm looking for and where I do my
best" section exists for the same reason and is written in Ted's own words,
first person, deliberately **not** third-party testimonials or pulled
LinkedIn recommendations — he wants the site to say who he is, not advertise
him.

## Structure, and why it's shaped this way

- **Nav is Home / About / Notes.** Research (the PhD work) is deliberately
  *not* in the top nav — it's linked from About as a supporting deep-dive,
  not the lead identity, because Ted is a practicing industry data scientist
  now, not an academic. Don't promote Research back to the nav without a
  reason; it was demoted on purpose.
- **No resume PDF.** Removed deliberately — Ted doesn't want to maintain a
  synced PDF. Site directs interested parties to make contact instead. Don't
  re-add a resume link without being asked.
- **Notes pulls itself.** The homepage's "Recent notes" section and the
  `/blog` index both loop over `site.posts` via Liquid — publishing a new
  post is enough, nothing else needs manual updating.
- **Visual design is intentionally deferred**, not neglected. See
  `DESIGN_NOTES.md` for what's queued (theme swap, mobile check, stale
  photos). Don't take on a design pass unless asked — the priority was
  fixing what the site *says* before what it *looks like*.

## Writing Notes posts

The framing on `/blog` is: *"less 'here's what I built,' more 'here's the
non-obvious call I had to make and what it taught me.'"* That's the default
lens, not a mandatory formula — **don't force every post into a
business-judgment lesson if that's not actually what the project is about.**
That was a real correction from Ted on the first draft of the Auralux post:
the initial draft picked two technical decisions and wrote them up like a
capability showcase, and it read as manufactured insight bolted onto a
project that wasn't actually about that. He wanted it cut down to what the
project genuinely was — something built for fun, because he likes cool,
colorful, glowy things — with one honest, specific thing he was proud of
(the menu being part of the experience, not a chore panel), not a forced
"and here's what this teaches us about engineering" angle.

So there are (at least) two legitimate shapes for a post, and the judgment
call is picking the right one for the actual project rather than defaulting
to one template:

1. **"Look at this fun/cool thing I built."** For personal/creative
   projects. The point is the idea and the fact that Ted spent real time on
   something whose only purpose is delight — not that it demonstrates a
   skill. Don't manufacture a lesson. If there's a genuinely interesting
   decision, mention it, but it's in service of the personality/craft
   angle, not a pretext for a business-judgment moral. Screenshots or
   visuals are a good fit here since the whole point is often "look at
   this."
2. **"Here's the non-obvious call and what it taught me."** For projects
   (work-adjacent or personal) where a real tradeoff was made — the kind of
   thing that shows how Ted thinks about ambiguous problems generally. This
   is the "connects technical detail to the big picture" mode from the
   site's core thesis. Structure: the problem in plain terms → the call
   that wasn't the obvious/default answer → what happened → a takeaway that
   generalizes past the specific project. There's a scaffold for this at
   `_drafts/TEMPLATE.md` — useful as a starting shape, not a checklist to
   fill mechanically.

There's also a category Ted mentioned but hasn't written yet: **learning
tools built for his own learning that others could use too** — likely a
third register, more teaching-oriented, not yet defined by an example. Use
judgment if one comes up; probably closer to mode 2 than mode 1, but check
in if unsure.

Voice notes, regardless of mode:
- First person, honest, a little self-deprecating when true ("the
  implementation isn't doing anything special" is a real line from the
  Auralux post — don't be afraid to say a project isn't technically
  impressive if the point is elsewhere).
- No corporate speak, no manufactured stakes.
- Keep it to a handful of paragraphs. These are Notes, not essays.
- The title should sound like something a person would actually say, not a
  headline (compare "Auralux: I just like things that glow" to the
  discarded draft title "Auralux: the best UI decision was deleting a
  button" — the second sounds like a Medium post, the first sounds like
  Ted).

## Before publishing

- Build locally to catch Liquid/markdown errors before pushing (see
  `README.md`/repo history for the Jekyll build workaround needed in some
  sandboxed environments — bundler can be broken there; installing the
  `jekyll` and `jekyll-theme-leap-day` gems directly and building without
  Bundler is the fallback).
- If real screenshots exist for a post, use them — but if you can't reliably
  get binary image data into a tool call intact (this bit us once with
  corrupted base64), don't guess-and-check silently. Ship the text, tell
  Ted what's missing and why, and let him decide whether to supply the
  image or wait.
