# Idea 2 local timing feasibility audit

The first step is to check whether localized timing after R can be measured consistently in the stored measured Frank XYZ beats. The earlier whole-beat alignment had a median shift of 0 ms and evaluated its shift with the same reference that selected it. This audit evaluates departures from the shift near R and uses a second, disjoint template for confirmation. The output is an engineering feasibility result with zero classifier fits. [Earlier alignment audit](record_tensor_quality_plan.md) · [Research backlog](RESEARCH_BACKLOG.md)

Use all 164 patients, 391 recordings and 55,863 denoised beats, with the original measured XYZ, labels and R positions. Partition each recording's chronological beat indices modulo four. For each group of query beats, use two different other groups to make separate median templates. Neither reference includes the query group, and the two references share no beats. All query beats contribute to summary denominators. Serially adjacent beats can still be correlated.

Use four fixed half-open windows in milliseconds relative to the cached R index. The anchor is [−50, 70), followed by [90, 180), [180, 270) and [270, 360). These are descriptive windows. Sweep each window between −24 and +24 ms in 2 ms increments. Score centered cosine similarity on the joint XYZ segment, using a fixed sample count and a single shift for all three channels. Positive lag means the observed segment occurs later. Channel centering and normalization are confined to the similarity calculation. Exact ties favor the smallest absolute shift and then the negative shift.

A lag is identifiable when both query and template have at least 0.005 mV centered RMS, the best cosine is at least 0.85, the maximum is inside the search range and its score exceeds any candidate at least 8 ms away by 0.002 or more. Agreement requires both disjoint references to pass and agree within 4 ms. A supported local shift also requires a repeatable near-R anchor, a same-sign departure of at least 8 ms from that reference's own near-R estimate, and at least +0.005 cosine improvement over the near-R-only shift when each estimate is evaluated against the other reference. This last comparison tests whether a local adjustment improves on a single global shift. It measures consistency, not causality or clinical timing accuracy.

## Controls and gate

- Inject known global translations of −16, −8, 8 and 16 ms in each recording median, with a common gain and fixed 0.001 mV noise. For each window, measure accepted coverage, recovery within 2 ms among accepted estimates and joint detection/recovery across all controls. The translated near-R and post-R lag difference should cancel.
- For the same reference medians test a zero-shift gain/noise case and a zero-shift amplitude deformation. The deformation changes a smooth part of the window's shape while keeping its time grid fixed. Count accepted apparent shifts of at least 8 ms. These probes characterize one family of confounds.
- A window passes the prespecified descriptive gate with at least 50% accepted positive-control coverage, at least 90% recovery among accepted controls, and at least 20% of patients with supported local shifts in at least 10% of beats using equal-record weighting. Controls test signal recovery and the gate tests prevalence. A pass supports a separate matched classifier test, subject to the shape ambiguity exposed by the controls.

Report counts, agreement, signed and absolute lag summaries, near-R-relative support and the cross-reference improvement for all declared windows. Pool rates equally across a patient's records, then summarize across patients. Report engineering results without using labels, fold outcome scores or patient exclusion. Raw waveforms and individual lag outputs remain local. Run a fresh process to reconstruct every beat and control and check the private array hashes before recording a completed audit.

```bash
.venv/bin/python scripts/run_local_timing_audit.py
```

The backlog's potential classifier comparison would keep the original waveform branch available and compare an unaligned representation, a bounded aligned representation, and aligned plus timing on the same patient folds. It is a distinct stage from proving that the estimator can detect a measurable signal.
