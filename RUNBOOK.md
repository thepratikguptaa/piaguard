# Panel runbook — 9 September 2026

Two things to know before planning anything else.

**The 17-slide cap is tighter than it looks.** The eight components add up to 15 slides at
their stated maxima. Add a title slide and a references slide and you are at 17, cover to
cover. There is no room for the table of contents, the certificates, or the Gantt chart
that were in the PBL-I deck. Those belong in the report, not the deck.

**Two components did not exist in PBL-I: Implementation (3 slides) and Results (3
slides).** They are 6 of the 17 — over a third of the presentation — and neither can be
filled with plans. That is the entire job this week.

---

## Slide map (17, cover to cover)

| # | Component | Content | Source |
|---|---|---|---|
| 1 | — | Title, team, supervisor | existing deck slide 1 |
| 2 | Abstract | Now past tense: what was built and what it measured | rewrite — see below |
| 3 | Introduction | Background, motivation, significance | existing slide 4, trimmed |
| 4 | Literature Review | Papers 1–3, with full publication details | existing slides 5–7 |
| 5 | Literature Review | Papers 4–5 + a synthesis row: gap → what we do about it | needs 2 more papers written up |
| 6 | Problem Identification | The four limitations, each tied to a cited gap | existing slide 8 |
| 7 | Solution Strategy | Four-layer architecture block diagram | README diagram |
| 8 | Solution Strategy | Class + use case diagram | existing slides 14–15 |
| 9 | Solution Strategy | Activity/sequence diagram + the formal detection rule | existing slides 16–17 + `detector.py` docstring |
| 10 | Implementation | Stack, dev methodology, module table | `README.md` layout table |
| 11 | Implementation | **Algorithm 1** — masked-span loss shift | below |
| 12 | Implementation | **Algorithm 2** — threshold calibration + fusion | below |
| 13 | Results | Table 1 (vs baselines) + calibrated operating point | `results/table1_main_results.csv` |
| 14 | Results | Fig 1 ROC + Fig 2 score separation | `results/fig1_roc.png`, `fig2_*.png` |
| 15 | Results | Table 3 per-family recall + Fig 5 latency + limitations | `table3_*`, `fig5_*`, README limitations |
| 16 | Paper Publication Details | Published or communicated — see the checklist below | **you must supply this** |
| 17 | — | References (IEEE) + thank you | existing slide 21 |

Slide 5 needs two more papers written up to satisfy "at least five research papers with
publication details". Your reference list already names candidates — Zou et al. on
transferable adversarial attacks, and the prompt injection attacks-and-defenses survey.
Each needs venue, year, DOI/arXiv id, inference, gap, and relevance, in the same table
format as slides 4.

Slide 16 is the one component with no technical work behind it and it is easy to forget.
The circular asks for details "for both (published or communicated)". If you have not
submitted anywhere, say so plainly and state the target: venue, submission window,
current draft status. A panel accepts "communicated to X, under review" or "targeting Y,
draft complete"; it does not accept a blank slide.

---

## Six-day plan

**Day 1 (Wed 3rd) — get real numbers.** Open `notebooks/PIAGuard_Colab.ipynb` on a T4.
Run the smoke test, then cell 3, and *look at the top spans*. If they land on the
injected instruction, the method works and everything else is downstream. If they land on
random tokens, you have a signal problem — the notebook's troubleshooting section covers
the order to check things in. Do not proceed to the full evaluation until cell 3 looks
right.

**Day 2 (Thu 4th) — evaluation.** Full run at `--target-fpr 0.01`. Then the model-size
comparison (`gpt2` vs `gpt2-medium`). Download `results/`. You now have Tables 1–4 and
Figures 1–5. Read them before you put them on a slide — you have to defend every number.

**Day 3 (Fri 5th) — strengthen the eval.** Two jobs. Add hard negatives to the clean set:
legitimate prompts that mention instructions, roles, security, system internals. These
are what separate a real FPR from a flattering one, and the panel will ask what your
false positives look like. Second, load a public benchmark via `--extra-csv` and report
that as the headline, with the seed set as the per-family breakdown.

**Day 4 (Sat 6th) — build the deck.** Slides 10–15 first, while the numbers are fresh.
Slides 2 and 5 next. Everything else is edited from PBL-I.

**Day 5 (Sun 7th) — report + rehearse.** The report carries what the deck cannot: full
literature tables, all four algorithm listings, the certificates, the Gantt chart, the
per-prompt appendix (`results/per_prompt_scores.csv`). Both deck and report need your
guide's signature — book that time now, not on the 8th.

**Day 6 (Mon 8th) — dry run and freeze.** Full run-through against the clock. Have the
demo ready as a fallback: `python scripts/demo.py --model gpt2` produces the per-layer
trace, and a live block of an injection lands harder than any bar chart. Have a recorded
version too — never rely on Colab connecting in the room.

---

## Slide-ready pseudocode

### Algorithm 1 — Masked-span loss shift detection

```
Input : prompt P, scoring LM M, window sizes W, aggregation eps
Output: anomaly score s, most suspicious span

1  T  ← tokenize(P)                              ; n ← |T|
2  L₀ ← meanNLL(M, T)                            ; baseline loss
3  V  ← ∅                                        ; masked variants
4  for each w ∈ W do
5      for i ← 0 to n−w step ⌈w/2⌉ do
6          V ← V ∪ { (T \ T[i : i+w], (i, i+w)) }
7  if |V| > maxPositions then V ← evenlySample(V, maxPositions)
8  for each batch B ⊆ V do                       ; batched, no gradients
9      L[B] ← meanNLL(M, B)
10 Δ  ← L₀ − L                                   ; Δᵢ > 0 ⇒ span i was costly
11 m  ← median(Δ) ; d ← 1.4826 · MAD(Δ)          ; robust scale
12 s  ← (max(Δ) − m) / (d + eps)                 ; self-normalising score
13 return s, span(argmax Δ)
```

Two lines carry the contribution and are worth saying out loud: **line 6**, because
masking one token of a phrase-length trigger leaves the trigger intact, and **line 12**,
because raw shift magnitude scales with prompt length so a global threshold cannot
transfer without the per-prompt normalisation.

### Algorithm 2 — Threshold calibration and fusion

```
Input : clean calibration set C (no attacks, disjoint from test set),
        target false positive rate α
Output: θ_block, θ_review

1  for each p ∈ C do
2      r_p ← sanitizerRisk(p)                     ; Layer 1 evidence
3      d_p ← lossShiftScore(p)                    ; Algorithm 1
4      f_p ← d_p + λ · r_p                        ; λ = 1.5
5  θ_block  ← quantile({f_p}, 1 − α)              ; α = stated FPR budget
6  θ_review ← quantile({f_p}, 0.85)
7  return θ_block, θ_review

At inference:  f > θ_block → BLOCK
               f > θ_review → REVIEW (degraded system prompt / human review)
               otherwise    → ALLOW
```

Line 1 is the defensible part: calibrating on clean prompts only, held out from the test
set, means θ is never fitted to the attacks it is evaluated against. If a panel member
asks one methodological question, it will be this one.

---

## Questions to have an answer for

- *Why not just fine-tune a classifier?* Cost and coverage. Fine-tuning needs labelled
  data per attack type and retraining as attacks evolve; this needs neither and works on
  any decoder-only model. Cite it as the gap the UniGuardian review identifies.
- *How did you pick θ?* You didn't — Algorithm 2 derives it from a clean set at a stated
  FPR budget. Show `run_metadata.json`.
- *What are your false positives?* Have three real examples from
  `per_prompt_scores.csv` where `label=0` and `verdict=BLOCK`, and a sentence on what
  they have in common.
- *Is your test set fair?* Say the honest thing: the seed set is small and hand-built,
  which is why the headline comes from a public benchmark and why the clean half includes
  deliberate hard negatives.
- *Which layer actually does the work?* Table 2. If the rule layer is carrying it, say so
  — that is a finding, not a failure.
- *Why score with GPT-2 and not the protected model?* Compute. Flag it as untested
  transfer and the obvious next experiment. Do not claim it works.
