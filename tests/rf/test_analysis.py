import pandas as pd

from rf.analysis import (
    ARTIFACT_SOURCE,
    BAND_RHO,
    GATES,
    VERDICTS,
    _in_band,
    band_endpoints,
    build_verdicts,
    is_go,
)
from rf.hypotheses import INFORMATIONAL, REGISTRY
from rf.hypotheses._artifacts import write_records

# the merges leave three verdicts without a driver of their own, so the redirect is load-bearing


class TestRedirects:
    def test_every_verdict_resolves_to_a_driver(self):
        for name in VERDICTS:
            source = ARTIFACT_SOURCE.get(name, name)
            assert source in REGISTRY, f"{name} reads {source}, which no driver writes"

    def test_absorbed_mechanisms_are_redirected(self):
        for name in ("label_nuisance_tradeoff", "se_calibration", "cumulant_margin",
                     "compressibility_budget"):
            assert name in VERDICTS
            assert name not in REGISTRY
            assert ARTIFACT_SOURCE[name] in REGISTRY

    def test_gates_are_scored_not_informational(self):
        for gate in GATES:
            assert gate in VERDICTS
            assert gate not in INFORMATIONAL


class TestBuildVerdicts:
    def test_missing_artifacts_yield_a_row_per_mechanism(self, tmp_path):
        out = build_verdicts(tmp_path)
        assert len(out) == len(VERDICTS)
        assert set(out["verdict"]) == {"missing"}

    def test_a_gap_in_one_mechanism_leaves_the_others_scored(self, tmp_path):
        # preemption can drop the rho band se_calibration reads; the other verdicts must survive it
        rows = [{"snr": 20, "measurement_snr": 20.0, "rho": r, "knee_measured": 0.4,
                 "knee_predicted": 0.4, "knee_err": 0.0, "knee_interior": True}
                for r in (0.3, 0.5)]
        write_records(tmp_path, "se_knee", 0, rows, "20dB")
        out = build_verdicts(tmp_path).set_index("mechanism")
        assert out.loc["se_calibration", "verdict"] == "incomplete"
        assert out.loc["se_knee", "verdict"] in {"accept", "reject"}
        assert out.loc["se_knee", "coverage"] == "1 seeds x 1 levels"

    def test_go_needs_both_gates(self):
        rows = [{"mechanism": m, "verdict": "accept", "detail": ""} for m in GATES]
        assert is_go(pd.DataFrame(rows))
        rows[0]["verdict"] = "reject"
        assert not is_go(pd.DataFrame(rows))


RHOS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9, 0.95]


def _band_rows(lo, hi, level):
    return [{"snr": 20, "measurement_snr": level, "rho": 0.5, "lo": lo, "hi": hi,
             "rho_dt": 0.46, "nonempty": lo <= hi, "eps": 0.0422, "eps_source": "prop1"}]


class TestOperatingBand:
    def test_endpoints_come_from_the_artifacts_not_a_constant(self, tmp_path):
        write_records(tmp_path, "admissible_band", 0, _band_rows(0.55, 0.65, float("inf")),
                      "noiseless")
        write_records(tmp_path, "admissible_band", 0, _band_rows(0.7, 0.95, 10.0), "10dB")
        bands = band_endpoints(tmp_path)
        assert bands == {float("inf"): (0.55, 0.65), 10.0: (0.7, 0.95)}

    def test_each_level_is_cut_to_its_own_band_inclusively(self, tmp_path):
        rows = [{"snr": 20, "measurement_snr": lv, "rho": r}
                for lv in (10.0, float("inf")) for r in RHOS]
        band = {10.0: (0.7, 0.95), float("inf"): (0.55, 0.65)}
        out = _in_band(pd.DataFrame(rows), band)
        # the endpoints themselves must survive: averaging identical floats used to drop them
        assert sorted(out[out["measurement_snr"] == 10.0]["rho"]) == [0.7, 0.8, 0.9, 0.95]
        assert sorted(out[out["measurement_snr"] == float("inf")]["rho"]) == [0.55, 0.6, 0.65]

    def test_no_artifacts_falls_back_to_the_scalar(self, tmp_path):
        assert band_endpoints(tmp_path) == {}
        rows = pd.DataFrame([{"snr": 20, "measurement_snr": 10.0, "rho": r} for r in RHOS])
        assert _in_band(rows, {})["rho"].min() >= BAND_RHO


class TestMeasurementWithoutAVerdict:
    def test_class_diameter_writes_artifacts_that_no_verdict_scores(self):
        # its pair statistics are recorded but no criterion is pre-registered over them, so the
        # driver must not appear on either side of the verdict wiring
        assert "class_diameter" in REGISTRY
        assert "class_diameter" not in VERDICTS
        assert "class_diameter" not in ARTIFACT_SOURCE.values()
