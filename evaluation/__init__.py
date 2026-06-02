"""Linear probe and visualization evaluation utilities."""

from evaluation.linear import (
    run_linear_probe,
    run_probe_suite,
    run_probe_suite_seeded,
    load_mel_embeddings,
)
from evaluation.visualizations import (
    plot_representation_umap_grid,
    load_mel_methods_umap_data,
    plot_mel_methods_umap,
    compute_psnr_alignment_sweep,
    plot_psnr_alignment_suite,
    plot_f1_ci_vs_ratio,
    find_centroid_tracks,
    build_showcase_data,
    plot_augmentation_showcase,
)

__all__ = [
    "run_linear_probe",
    "run_probe_suite",
    "run_probe_suite_seeded",
    "load_mel_embeddings",
    "plot_representation_umap_grid",
    "load_mel_methods_umap_data",
    "plot_mel_methods_umap",
    "compute_psnr_alignment_sweep",
    "plot_psnr_alignment_suite",
    "plot_f1_ci_vs_ratio",
    "find_centroid_tracks",
    "build_showcase_data",
    "plot_augmentation_showcase",
]
