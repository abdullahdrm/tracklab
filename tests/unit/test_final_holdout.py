from __future__ import annotations

import pytest

from evaluation.final_holdout import HOLDOUT_SEQUENCES, _load_protocol


def test_frozen_protocol_uses_only_holdout_sequences() -> None:
    protocol = _load_protocol("configs/day17_final_protocol.yaml")
    assert tuple(protocol["sequences"]) == HOLDOUT_SEQUENCES
    assert {case["preprocessing"] for case in protocol["cases"]} == {
        "none",
        "clahe",
        "ycrcb_clahe_gamma",
    }


def test_frozen_protocol_rejects_tuning_sequence(tmp_path) -> None:
    protocol = """\
split: holdout
sequences: [MOT17-04-FRCNN, MOT17-10-FRCNN, MOT17-13-FRCNN]
cases: [{id: bad, preprocessing: none, pipeline_config: configs/pipeline.yaml, description: bad}]
"""
    path = tmp_path / "bad.yaml"
    path.write_text(protocol, encoding="utf-8")
    with pytest.raises(ValueError, match="exactly MOT17-02/10/13-FRCNN"):
        _load_protocol(path)
