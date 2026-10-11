# SHL Grammar Scoring Engine for Spoken English

Solution to the SHL Research Engineer Kaggle challenge (*SHL Hiring Assessment 2026*): predict a continuous
MOS grammar score (0–5) for 45–60 s speech clips. 769 training clips, 216 test clips, evaluated with Pearson correlation and RMSE.

| | Pearson | RMSE |
|---|---|---|
| 5-fold cross-validation (out-of-fold, test-length-weighted, nested stacker) | 0.854 | 0.499 (0.501 averaged over 5 stacker splits) |
| Training data RMSE (final ensemble, all 732 training clips) | 0.872 | 0.496 |
| Kaggle public leaderboard | | **0.3417** |

The training RMSE is close to the CV RMSE because the stacker is fitted on the base models' out-of-fold predictions;
the in-sample training RMSE of the individual refittable models (0.16–0.39) is reported in notebook §10.

## Approach

```
audio (16 kHz)
 ├─► Whisper turbo ASR (verbatim prompt, word timestamps) ─► transcript
 │      ├─► DeBERTa-v3-base / -large, fine-tuned (regression) ──────────────┐
 │      └─► grammar & fluency features + mpnet embedding ─► Ridge/SVR/LGBM ──┤
 ├─► WavLM-large (frozen, best layers)                                      │
 │      ├─► clip mean (+ std) ─► Ridge / SVR ────────────────────────────────┤
 │      └─► 80 ms frame sequence ─► attention-pooling head ───────────────────┤─► non-negative linear stacker
 └─► Whisper-large-v3 encoder (frozen, best layers)                         │     ─► per-length bias correction
        ├─► clip mean (+ std) ─► Ridge / SVR ────────────────────────────────┤     ─► score
        └─► 80 ms frame sequence ─► attention-pooling head ───────────────────┘
```

## Key findings
1. **Artificial score-0 batch.** The 37 score-0 clips (ids 5037–5073) are ~10 dB louder, contain no silence and are absent from the
   test set. Removed from training.
2. **Train/test length shift.** 69 % of test clips are short (<50 s) vs 24 % of training clips, and short clips score 0.46 lower.
   Handled with importance weights in every loss/metric and a per-length bias correction.
3. **Speech is as informative as text.** Frozen speech-model features match a fine-tuned DeBERTa on transcripts; their errors are
   only moderately correlated, so combining them gave the largest gains.
4. **Keeping the time dimension of the speech helped most at the end.** Small attention heads on 80 ms frames of the frozen
   speech models, instead of one averaged vector per clip, moved the leaderboard from 0.3510 to 0.3417.
5. **The test set contains prompts never seen in training** (e.g. floods: 0 training / 22 test clips). With whole topics held out,
   the text models lose 0.035–0.058 RMSE while the speech models do not degrade, which is why speech-heavy stacks transfer better.
6. **Negative results worth knowing.** A Qwen2.5-7B "LLM examiner" given the official rubric correlates only 0.41 (zero-shot);
   end-to-end fine-tuning of WavLM on 8 s crops did not beat frozen features; a third encoder (w2v-BERT 2.0) improved CV but
   not the leaderboard. DeBERTa seed variance is large (CV RMSE 0.562–0.591), so small CV gains were treated as noise.
7. **Remaining error is mostly rater inconsistency.** Broken Whisper transcripts (loops, long silences) explain only a few percent
   of the error; the worst errors are clips whose text and audio look like typical 3–3.5 answers but were rated 5 (or 2).

## Repository
| File | Content |
|---|---|
| `shl_grammar_scoring.ipynb` | Full pipeline, report, evaluation, visualisations (executed, with outputs) |
| `submission.csv` | Final test-set predictions (Kaggle public LB 0.3417) |
| `requirements.txt` | Python packages |

## Running
The notebook runs on Kaggle with the competition data attached.
* `RUN_FULL = False` (default): loads the expensive cached artefacts (transcripts, DeBERTa and attention-head out-of-fold
  predictions, frozen speech-model features) from the attached datasets and recomputes everything else (~15 min).
* `RUN_FULL = True`: recomputes every stage from raw audio (~4 h on a T4 GPU). Requires `pip install openai-whisper`.

The competition audio and CSV files are not included in this repository.
