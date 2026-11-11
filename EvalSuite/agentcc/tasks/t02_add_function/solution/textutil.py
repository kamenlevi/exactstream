import re


def title_case(s):
    return ' '.join(w.capitalize() for w in s.split())


def slugify(text):
    return re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')
