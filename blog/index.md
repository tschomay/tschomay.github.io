---
layout: page
title : Notes
---
## Notes
<p align="justify">
Short write-ups on projects I'm working on — less "here's what I built," more "here's the non-obvious call I had to make and what it taught me."
</p>

<ul>
  {% for post in site.posts %}
    <li>
      <a href="{{ post.url }}">{{ post.title }}</a> <small>— {{ post.date | date: "%b %-d, %Y" }}</small>
    </li>
  {% endfor %}
</ul>
