# flow_matching with conditional models for discrete label maps

Research modifications of Meta's [flow_matching](https://github.com/facebookresearch/flow_matching) library
(Lipman et al., *Flow Matching Guide and Code*, 2024; [arXiv:2412.06264](https://arxiv.org/abs/2412.06264)). The upstream CC BY-NC 4.0 license is retained in `LICENSE`; the original README is kept as `README_upstream.md`.

## Overview

The library implements continuous and discrete flow matching, with the model architectures living under
`examples/`. This repository adds an installable `flow_matching.model` subpackage with **conditional architectures for
discrete flow matching on a pixel grid**: the state is a map of per-pixel discrete labels, and generation is conditioned
on an observed image.

Research work from 2025. The training and evaluation scripts are not provided.

## What was changed

- **`flow_matching/model/`** (new). The example architectures (`unet.py`, `discrete_unet_init.py`, `transformer.py`,
  `rotary.py`, `ema.py`, `nn.py`) were moved from `examples/` into the package so that external code can import them.
  `nn.normalization` now chooses a GroupNorm group count that divides small channel widths.
- **`flow_matching/model/cnn.py`**: `DFM_CNN`, a FiLM-conditioned residual CNN that outputs per-pixel logits over the
  label vocabulary, with the conditioning image concatenated as an extra input channel and a configurable kernel size.
  `WrappedModel` adapts it to `MixtureDiscreteEulerSolver`, passing the conditioning image through `model_extras` and
  repeating it for several samples per condition.
- **`flow_matching/model/discrete_unet.py`**: `ConditionalDiscreteUNetModel`, a pixel-token embedding followed by the
  upstream UNet, with a 1×1-projected conditioning image concatenated through the UNet's `concat_conditioning` input;
  its own `WrappedModel` for the solver.

## Status

Research snapshot; not actively maintained.

## License

CC BY-NC 4.0, as upstream (non-commercial use only). Original code copyright Meta Platforms, Inc. Files under
`flow_matching/model/` that carry Meta's header were copied or adapted from the upstream examples; additions by Andrej Leban, 2025.
