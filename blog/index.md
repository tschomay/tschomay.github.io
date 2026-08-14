---
layout: page
title: Notes
---

<p align="justify">
Short write-ups on projects I'm working on — less "here's what I built," more "here's the non-obvious call I had to make and what it taught me."
</p>

<div class="note-cards" markdown="0">
{% for post in site.posts %}
  <div class="note-card">
    <span class="when">{{ post.date | date: "%b %-d, %Y" }}</span>
    <div>
      <h3 class="ttl"><a href="{{ post.url }}">{{ post.title }}</a></h3>
      <p>{{ post.excerpt | strip_html | truncate: 160 }}</p>
      {% if post.topic %}<div class="tags"><span class="tag">{{ post.topic }}</span></div>{% endif %}
    </div>
  </div>
{% endfor %}
</div>
