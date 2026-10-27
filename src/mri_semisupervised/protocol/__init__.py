"""The evaluation protocol.

Everything that touches a label lives here, and everything here respects one rule: the
test fold is read once, at the end, to produce predictions. It takes part in no decision —
not the choice of clustering method, not the cluster-to-class alignment, not the
checkpoint, not the decision threshold.

The rule is not a comment. `tests/integration/test_no_leakage.py` runs the protocol with a
recorder in place of the training and asserts that no test image reaches an inner split or a
pseudo-label set. `tests/unit/protocol/test_pseudo_labels.py` flips the labels the alignment
mask hides and asserts that the alignment does not move.
"""
