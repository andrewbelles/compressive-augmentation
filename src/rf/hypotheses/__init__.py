from . import (
    operator_isometry,
    nuisance_equivariance,
    dictionary_compressibility,
    se_calibration,
    backprojection_law,
    operator_draw_variance,
    cumulant_margin,
    label_nuisance_tradeoff,
    admissible_band,
)

# registry of first-stage hypothesis drivers, keyed by mechanism

REGISTRY = {
    "operator_isometry": operator_isometry.run,
    "nuisance_equivariance": nuisance_equivariance.run,
    "dictionary_compressibility": dictionary_compressibility.run,
    "se_calibration": se_calibration.run,
    "backprojection_law": backprojection_law.run,
    "operator_draw_variance": operator_draw_variance.run,
    "cumulant_margin": cumulant_margin.run,
    "label_nuisance_tradeoff": label_nuisance_tradeoff.run,
    "admissible_band": admissible_band.run,
}

# drivers that need no dataset (operator algebra only)
DATA_FREE = {"operator_isometry"}

# controls and theory calculations: they validate the implementation, not the augmentation
INFORMATIONAL = {"operator_isometry", "backprojection_law", "admissible_band"}
