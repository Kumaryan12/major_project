# Upstream HYDRA multivariate transform

Authors: Angus Dempster, Daniel F. Schmidt, Geoffrey I. Webb.

Source: https://github.com/angus924/hydra/blob/144bb7aa3186654042f00f36b078370c620acb06/code/hydra_multivariate.py

Revision: `144bb7aa3186654042f00f36b078370c620acb06` (retrieved 16 November 2026).
`hydra_multivariate.py` is unchanged from that revision. The upstream GPL-3.0
license is included in `LICENSE`; this attribution/license applies to the
vendored code. The original authors explicitly mark this multivariate extension
experimental and untested. Project tests validate its actual use here, without
turning it into an independently validated medical method.

Paper: https://arxiv.org/abs/2203.13652

The bounded study wraps it with scoped seeds, CPU chunks, training-only common
RMS normalization, patient aggregation and a matched StandardScaler/ridge
classifier. The wrapper does not use upstream SparseScaler. No new numerical
dependency or binary replacement is required.
