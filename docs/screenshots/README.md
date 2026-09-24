# Screenshots for the engineering report

Both images are referenced from the report section of `../../README.md` and must
be captured from the **deployed Streamlit app**, not a local run.

| File | How to capture |
|------|----------------|
| `converged-lr-0.01.png` | Train tab, default dataset (slope 2.5, intercept 1.0, noise 2.0, n = 500), lr = **0.01**, batch size 32, max epochs 200, early stopping on. Capture after the run finishes so the loss curve, the "converged after 34 epochs" message, the metric tiles and the fitted-line overlay are all visible. |
| `diverged-lr-0.035.png` | Same dataset, lr = **0.035**. Use 0.035, not a larger value: at lr = 1.5 the loss overflows to infinity during the first epoch, so there is no curve to show — only the "no curve to draw" caption. |

A third shot of the Run History tab with both runs selected in "Compare loss
curves" is optional but makes the comparison paragraph concrete.
