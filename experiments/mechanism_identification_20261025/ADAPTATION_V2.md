# V2 adaptation, registered after V1 and before V2 results

V1's original global-photo budget accepts only 2/8 known legal endpoints.
Local-witness calibration accepts 8/8, but expanded-bank model selection is
correct only 4/8: identical-material pairs expose left/right binding ambiguity.
None of the three oracle-fitted low-dimensional families meets the predeclared
reference tolerance. This is evidence about these families, not all mechanisms.

V2 freezes the following changes before looking at new scenes 16--31:

- Add a positive multiplicative material-gain family. Eight source-chromaticity
  response functions times eight fixed degree-7 Bernstein x-coordinate functions
  form a partition of unity. A common 192 coefficients (three output channels)
  lie in [.05,12]. No ground-truth object support or text mask defines the basis.
  As with prior source-coordinate appearance families, this is an image-level
  photometric surrogate; it is never multiplied into the latent editing field.
- Also fit graph rank 32 (192 parameters), to distinguish parameter count from
  source-colour/coordinate structure. Fitting uses oracle target pixels solely
  to measure coverage, not to claim a usable candidate generator.
- Add full-prompt assignment contrast (target versus its left/right-swapped
  target) and its flip-antisymmetric counterpart. The synthetic prompt grammar
  gives the swapped text exactly; this is NOT general free-text parsing.
- Evaluate those readouts separately. Calibrate their gain floors from scenes
  4--7 only, using the same half-median-positive-gain rule. Do not use tiny
  normalized directions or change floors based on test outcomes.
- New dataset: scenes 0--7 retain development/calibration roles, 8--15 are now
  adaptation data, 16--31 are fresh renderer-held-out scenes. The generator and
  split are fixed before V2. These remain one restricted renderer, not unseen
  natural-image editing tasks.
- Repeat the observation x candidate-bank cross, and run source VAE projection
  controls after mechanism coverage improves. Do not launch full FlowEdit edits
  merely because an oracle-target fit succeeds.
