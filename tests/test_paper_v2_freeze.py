import pytest

from paper_v2.freeze import _clean_flip, _prompt_records


def test_clean_flip_accepts_nonconflicting_one_prover_resolution():
    a = {"consensus": False, "resolution": "one_prover_only"}
    b = {"consensus": True, "resolution": "dual_agreement"}
    assert _clean_flip(a, b)


def test_clean_flip_rejects_unresolved_or_conflict():
    assert not _clean_flip(
        {"consensus": None, "resolution": "unresolved"},
        {"consensus": True, "resolution": "dual_agreement"},
    )
    assert not _clean_flip(
        {"consensus": False, "resolution": "conflict"},
        {"consensus": True, "resolution": "dual_agreement"},
    )


def test_prompt_hashes_are_stable_and_side_specific():
    first = _prompt_records("pair", "prompt a", "prompt b")
    second = _prompt_records("pair", "prompt a", "prompt b")
    assert first == second
    assert first[0]["prompt_hash"] != first[1]["prompt_hash"]


def test_frozen_directory_cannot_be_overwritten(tmp_path):
    from paper_v2.freeze import freeze_pilot

    output = tmp_path / "frozen"
    output.mkdir()
    with pytest.raises(FileExistsError):
        freeze_pilot(output)
