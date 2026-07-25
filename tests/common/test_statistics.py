import random

import torch

from common.statistics import scatter_ratio

D = 4
PER_CLASS = 50


def _clustered(device, spread, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    eye = torch.eye(D, device=device)
    return torch.cat([
        torch.randn(PER_CLASS, D, generator=g, device=device) + spread * eye[c]
        for c in range(D)
    ])


def _labels():
    return [c for c in range(D) for _ in range(PER_CLASS)]


class TestScatterRatio:
    def test_separated_clusters_exceed_one(self, device):
        assert scatter_ratio(_clustered(device, 8.0), _labels()) > 5.0

    def test_shuffled_labels_destroy_separation(self, device):
        # null: the same features carry no separation once labels are permuted
        shuffled = _labels()
        random.Random(0).shuffle(shuffled)
        assert scatter_ratio(_clustered(device, 8.0), shuffled) < 0.5

    def test_overlapping_clusters_score_lower(self, device):
        wide = scatter_ratio(_clustered(device, 8.0), _labels())
        tight = scatter_ratio(_clustered(device, 0.5), _labels())
        assert tight < wide

    def test_single_class_is_nan(self, device):
        out = scatter_ratio(_clustered(device, 1.0), [0] * (D * PER_CLASS))
        assert out != out
