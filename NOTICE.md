ORSDet

ORSDet is cleaned reproducibility code for the SKAO SDC1 oriented radio source
detector. It includes a bundled CIANNA C/CUDA backend source tree under src/.

Upstream CIANNA notice:

Convolutional Interactive Artificial Neural Networks by/for Astrophysicists
(CIANNA) Code

These files are Copyright (C) 2024 David Cornu
(https://github.com/Deyht/CIANNA), but released under the Apache License,
Version 2.0.

ORSDet modifications and additions:

Copyright (C) 2026 ORSDet contributors.

The ORSDet release adds and modifies code for the SKAO SDC1 oriented radio
source detection workflow, including cleaned train/test entry points, ORSDet
data preparation, oriented-box target construction, detection, non-maximum
suppression, flux refinement, evaluation utilities, packaging, documentation,
and the released checkpoint under weights/.

The bundled CIANNA source tree under src/ may include ORSDet packaging and
runtime integration changes. See src/MODIFICATIONS.md.

Official SKAO SDC1 raw data files are not redistributed in this repository.
Users must obtain those files separately and comply with their applicable
terms.
