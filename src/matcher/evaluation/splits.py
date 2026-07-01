"""Frozen data splits — shared so every approach is measured on identical data."""

RANDOM_SEED = 42  # use everywhere a split or seed is needed


def load_fit_splits():
    """Return (train, test) preserving the fit dataset's NATIVE split — do not re-split.

    Phase-1 success criterion: the cnamuangtoun/resume-job-description-fit dataset
    ships its own train/test split; load it as-is.

    TODO(A): implement via matcher.data.loaders.load_fit_dataset().
    """
    raise NotImplementedError("Person A: return the dataset's native (train, test) split")
