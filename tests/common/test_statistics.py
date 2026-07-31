import math
import random

import torch

from common.statistics import class_margins, scatter_ratio

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


class TestClassMargins:
    def test_margins_recover_the_planted_separation(self, device):
        # the class means sit on scaled basis vectors, so the pair distances are known in closed form
        lo, q10 = class_margins(_clustered(device, 8.0), _labels())
        truth = 8.0 * math.sqrt(2.0)
        assert abs(lo - truth) < 1.0
        assert q10 >= lo

    def test_collapsed_classes_shrink_the_margin(self, device):
        wide, _ = class_margins(_clustered(device, 8.0), _labels())
        tight, _ = class_margins(_clustered(device, 0.5), _labels())
        assert tight < wide

    def test_single_class_has_no_margin(self, device):
        lo, q10 = class_margins(_clustered(device, 1.0), [0] * (D * PER_CLASS))
        assert lo != lo and q10 != q10
