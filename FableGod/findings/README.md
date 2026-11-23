# Findings: small, dated notes, published at once

Kamen, Wed 4 Nov 2026: every measurement, test and idea is documented on GitHub. When something looks new, a short note
goes out first, and the details follow later.

## When to write one

Write a note when an idea has both:
- a first number (measured [K], or arithmetic on measured numbers [P]);
- a prior-art status: NEW, adapted, or unchecked.

You don't need to wait for the full verdict. Write it, then tell the orchestrator.
- A possible NEW find: the orchestrator runs `publish_now.sh` at once.
- Everything else (adapted / published): it goes out with the hourly run at :05.

## File name

`YYYY-MM-DD_HHMM_<short-slug>.md` in this folder (`~/FableGod/findings/`). Take the time from `date`.

## Shape (one page at most)

```
# <the claim in one line>
- Date: YYYY-MM-DD HH:MM EET. Session: <name>.
- Claim: <one or two sentences>.
- Number: <value, units, conditions (RAM line, cache, prompts), EXACT or FAST>. Mark [K] / [P] / [G].
- How: <script and commit hash, data file>.
- Prior art: NEW / adapted (<closest paper>) / unchecked.
- Status: <first measurement / independent re-derivation / map PASS / same-scores PASS>.
- Next: <the one test that would confirm it>.
```

## Keep out of every note

- Personal details: names other than the project's, the owner's schedule, family.
- Account names, emails, keys and tokens.

The public build redacts some of these and blocks the whole publish on any leak-check hit.
