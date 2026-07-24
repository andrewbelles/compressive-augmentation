from . import (
    operator_isometry,
    nuisance_equivariance,
    gabor_compressibility,
    se_calibration,
    structure_vs_unstructured,
    cumulant_margin,
    admissible_band,
)

# registry of first-stage hypothesis drivers, keyed by mechanism

REGISTRY = {
    "operator_isometry": operator_isometry.run,
    "nuisance_equivariance": nuisance_equivariance.run,
    "gabor_compressibility": gabor_compressibility.run,
    "se_calibration": se_calibration.run,
    "structure_vs_unstructured": structure_vs_unstructured.run,
    "cumulant_margin": cumulant_margin.run,
    "admissible_band": admissible_band.run,
}

# drivers that need no dataset (operator algebra and band synthesis)
DATA_FREE = {"operator_isometry", "admissible_band"}
