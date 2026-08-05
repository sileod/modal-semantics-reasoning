import pytest

from paper_v2.semantics import (
    FRAME_PROPERTIES,
    Semantics,
    changed_axes,
    load_config,
    validate_config,
    validate_semantic_difference,
)


def test_config_is_valid():
    config = load_config()
    assert set(config["axes"]) == {"frame", "domain", "designation"}
    assert config["benchmark"]["target_pairs_per_axis"] == 400


def test_config_rejects_mismatched_scale():
    config = load_config()
    config["final"]["target_pairs"]["frame"] = 399
    with pytest.raises(ValueError, match="axis targets"):
        validate_config(config)


@pytest.mark.parametrize(
    ("left", "right", "property_name"),
    [
        ("K", "D", "serial"),
        ("K", "T", "reflexive"),
        ("T", "B", "symmetric"),
        ("T", "S4", "transitive"),
        ("B", "S5", "transitive"),
    ],
)
def test_frame_contrasts_change_one_property(left, right, property_name):
    assert FRAME_PROPERTIES[left] ^ FRAME_PROPERTIES[right] == {property_name}
    a = Semantics.make(left)
    b = Semantics.make(right)
    validate_semantic_difference("frame", (left, right), a, b)


def test_domain_contrast_has_fixed_background():
    a = Semantics.make("D", domain="varying")
    b = Semantics.make("D", domain="cumulative")
    assert changed_axes(a, b) == {"domain"}
    validate_semantic_difference("domain", ("varying", "cumulative"), a, b)


def test_designation_contrast_has_fixed_background():
    a = Semantics.make("D", designation="rigid")
    b = Semantics.make("D", designation="flexible")
    assert changed_axes(a, b) == {"designation"}
    validate_semantic_difference("designation", ("rigid", "flexible"), a, b)


def test_extra_semantic_change_is_rejected():
    a = Semantics.make("D", domain="varying")
    b = Semantics.make("D", domain="cumulative", designation="flexible")
    with pytest.raises(ValueError, match="change only domain"):
        validate_semantic_difference("domain", ("varying", "cumulative"), a, b)


def test_side_orientation_may_be_reversed():
    a = Semantics.make("T")
    b = Semantics.make("K")
    validate_semantic_difference("frame", ("K", "T"), a, b)
