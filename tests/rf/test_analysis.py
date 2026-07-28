import pandas as pd

from rf.analysis import ARTIFACT_SOURCE, GATES, VERDICTS, build_verdicts, is_go
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
