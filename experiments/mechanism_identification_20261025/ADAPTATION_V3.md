# V3 numerical coverage audit and latent feasibility

V2's material family reaches the reference tolerance on 12/16 fresh scenes;
rank-32 graph coefficients (same count 192) reach 0/16. This supports improved
coverage on this renderer, not an automatic text-conditioned editing claim.
Assignment-only readouts fail; replacing completion evidence with a small
assignment margin allows partial edits. Test conjunctions separately, preserving
the original completion constraint. Those conjunctions are adaptive analysis,
not independent held-out confirmation.

Before interpreting the four material misses as a capacity limit, audit fitting:
V2 forms downsample(source)*downsample(basis), which is not generally the actual
downsample(source*basis). In addition, its bounded quadratic solver hits limits.
V3 fixes the numerical fitting objective, without changing the material family,
coefficient bounds, or reference tolerance. Build the exact native-pixel normal
equations, factor the convex quadratic, and solve its small bounded least-squares
problem. Record KKT residual, native-pixel loss and finite decoded images.

Use scenes 4--7 for calibration, 16--31 as already inspected diagnostics, and
32--39 as fresh renderer-held-out controls registered before their results.
V3 data generation retains earlier factors unchanged. No source truth is given
to model readouts; oracle fits are only family-coverage upper bounds.

For preselected scenes 32--35, encode source and legal target with the original
frozen VAE. Test target reconstruction, the finite decoded chord, a source-fixed
material edit lifted using paired encodings, its strict identity control, and
decoder JVP against actual finite differences. Report errors against matched VAE
target separately from original rendered target. A reachable oracle endpoint
does not establish that text/velocity automatically discovers that direction.

No full SD3 trajectory until observation/action gates have reliable evidence.
