# Engineering decisions and omissions

This page covers the choices that could reasonably have gone the other way. The protocol itself
is described in [`protocol.md`](protocol.md).

## Where labels are read

```
src/mri_semisupervised/
├── data/         identity from content, inventory, preprocessing
├── features/     frozen ResNet50 embeddings, cached and guarded
├── models/       clustering helpers, the shared classifier
├── protocol/     splits, pseudo-labels, arms, metrics, intervals, orchestration
└── viz/          plotting helpers
```

The most important boundary runs between `models/` and `protocol/`. In the first version, the
clustering module `models/clustering.py` also scored each clustering against the labels and
aligned clusters to classes, so the leak sat in what looked like an ordinary modelling call.
Every step that uses a label to choose a clustering, a pseudo-label, a checkpoint or a threshold
now lives under `protocol/`, where a reviewer can find it: the splits, the pseudo-labels, the
arms, the metrics, the intervals and the orchestration. The `data/` package records the label of
each file in the inventory, and it stops the build when one image carries two different labels.

## Frozen and cached embeddings

The backbone is a ResNet50 with ImageNet weights, used as a feature extractor with its weights
frozen. Each file is embedded once and the result is cached as a Parquet file, because the same
embeddings feed every clustering candidate of every fold. Recomputing them for each fold would
repeat the same pass over 1 506 files twenty-five times.

The cache records the backbone it was built with and whether histogram equalisation was
applied. `scripts/build_features.py` refuses to reuse a cache built with a different backbone or
preprocessing, or over images that do not match the ones on disk. The experiment itself reads
the Parquet file directly and picks the plain or the equalised cache by file name. An equalised
cache read by a plain run would silently change the embeddings behind every downstream number,
so equalisation writes to a separate cache file.

## A small classifier

The classifier is a ResNet18 fine-tuned from ImageNet weights. Every arm starts from the same
ImageNet backbone, while the weights of the final two-class layer depend on a per-arm seed. In
the main run, the three arms that pre-train on clustering pseudo-labels all take 246
pre-training steps. Fine-tuning stops early on the inner validation split, so its length varies
with the arm and the fold. The classifier matters least in this design, because the experiment
tests the pre-training. It is therefore the same in every arm, and small enough that the 25
folds of all eight arms ran in under two hours on one consumer GPU.

## Two PyTorch indexes

The experiments train on an NVIDIA card under Windows, so `torch` and `torchvision` resolve
against the CUDA 12.6 index there. Continuous integration runs on Linux without a GPU, where
the CUDA wheels would add about three gigabytes of NVIDIA libraries to a test suite that runs on
the CPU. The index is selected by platform in `pyproject.toml`, since `uv` resolves the lock
file before installation and cannot detect the graphics card.

## Generated notebooks

`scripts/build_notebooks.py` writes all five `.ipynb` files. The text of every cell therefore
sits in a Python file that is versioned and readable in a diff, and the notebooks import the
same `mri_semisupervised` package as the scripts. `scripts/run_notebooks.py` executes each
notebook from a clean kernel, so the stored outputs all come from one execution.

## Alternatives deliberately left out

**Self-supervised pre-training on the pool.** SimCLR or a masked autoencoder trained on the
1 300 unlabelled slices is the approach with the strongest case on this data. It is out of scope
because the question here is whether pseudo-labels, taken from a clustering or from the
classifier's own predictions, improve the classifier.

**Recalibration.** The probability scale is measurably off, and it is left uncorrected. With 20
labelled validation images per fold, a Platt or isotonic map would mostly fit its own noise, so
the size of the calibration gap is published as measured.

**Histogram equalisation by default.** It adds 0.013 to the supervised ROC AUC, and it lowers
the adjusted Rand index (ARI) between the clustering and the labels from 0.457 to 0.205. The
clustering is the object under study, so the default pipeline leaves the pixels as they are.
Both runs are published, and the reader can compare them.

**Patient-level grouping.** The archive carries no patient identifier, so the folds group by
image and grouping by patient is impossible. [`data-source.md`](data-source.md) describes what
that limitation prevents.

**A service, a container or a user interface.** The repository delivers a measurement and the
evidence behind it, and it does not recommend deploying the model. A dashboard over 99
evaluation images would add little to the tables already published.
