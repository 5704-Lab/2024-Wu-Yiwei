# V5: close a natural-image mechanism-coverage gap

V4's multiplicative model misfits the known geometry-preserving global-white
control (.65*source+.35), particularly at dark pixels. Its unexplained phase
then becomes large although the original phase change is small. This is a
mechanism-coverage failure, not evidence of geometric damage.

Before V5 results, add only three global additive coefficients to the existing
192 positive material-gain coefficients. Gain bounds stay [.05,12]; global
offset bounds are [-.25,.75]. The fixed white control is now representable
exactly by gain=.65 and offset=.35. Solve the same exact native-pixel convex
fit and retain KKT audits. No target ROI, velocity gating or new model is added.

Compare old/new families on the four natural sources and known photometry,
mirror, translation and retained real SD3 failures. The photometries test image
geometry coverage, not whether the original local edit instruction permits them.
For controlled scenes, calibration remains scenes 4--7; use new scenes 40--47
for this revised mechanism's observations, before viewing their results.

Repeat RGB/unexplained-phase thresholds separately; do not claim phase-specific
benefit if RGB already rejects the same cases. Also perform direct-bank selection
without the oracle-fitted legal candidate. V4's fitted-candidate 8/8 is an oracle
bank result; the no-fitted-candidate audit is 7/8, including a wrong-object edit.
Neither constitutes automatic SD3 editing success.
