---
layout: home
title: Ted Schomay
role: Principal Data Scientist
thesis: What you do is only part of the picture — it's how you do it that has the bigger impact.
portrait: /assets/images/Ted2.jpg
---

## Welcome
<p align="justify">
I'm a Principal Data Scientist, but the technical answer has rarely been the hard part of my job. The hard part is connecting that answer to what the business actually needs, and doing it in a way that makes people want to keep working with you.
</p>

<p align="justify">
I am experienced across biomedical research, finance, and marketing technology, building models and products from early concept through to production. The work I'm most proud of, though, is usually the rescue — stepping into a product that's already shipped and quietly not delivering, finding the gap nobody's named yet, and reconnecting the technical fix to what the business actually needed from it in the first place. Thanks for checking out my page — have a look around, or reach out if you'd like to talk more.
</p>

## What I'm looking for and where I do my best
<p align="justify">
I do my best work in places where I'm expected to talk to both the whiteboard and the roadmap — where a data science team needs someone to turn a model into a decision an executive can act on, not just a metric in a slide. I'm drawn to ambiguous problems: the ones where the hardest call isn't how to build something, but whether to build it, and what tradeoff you're accepting either way.
</p>

<p align="justify">
I care as much about the team around the work as the work itself. I've sought out and created opportunities to build culture where I've worked — leading lunch-and-learns and hackathons, mentoring and onboarding new teammates, and helping shift at least one team from a burnout cycle into a supportive, productive one. I'm an optimist who assumes good intent, and I'd rather build a bridge between two disagreeing teams than pick a side.
</p>

## Recent notes
<ul class="notes-list">
{% for post in site.posts limit: 3 %}
  <li>
    <span class="d">{{ post.date | date: "%b %-d, %Y" }}</span>
    <span class="t"><a href="{{ post.url }}">{{ post.title }}</a></span>
    <span>{% if post.topic %}<span class="tag">{{ post.topic }}</span>{% endif %}</span>
  </li>
{% endfor %}
</ul>
<p><a href="/blog" class="btn">See all notes →</a></p>

## Contact
<div class="contact-row">
<span id="email"></span>
<a href="https://www.linkedin.com/in/tschomay" target="_blank" rel="noopener" class="btn">LinkedIn ↗</a>
</div>
<script type="text/javascript">
(function(){
  var user = "tschomay", domain = "gmail.com";
  var addr = user + "@" + domain;
  document.getElementById("email").innerHTML = "<a href='mailto:" + addr + "' class='btn solid'>" + addr + "</a>";
})();
</script>
<noscript>Email link requires Javascript. Message me on LinkedIn instead.</noscript>
