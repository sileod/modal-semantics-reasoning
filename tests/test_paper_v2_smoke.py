from paper_v2.smoke import _enabled_axes


def test_smoke_skips_disabled_axes():
    config = {
        "axes": {
            "frame": {"contrasts": [["K", "D"]]},
            "designation": {"enabled": False, "contrasts": [["rigid", "flexible"]]},
        }
    }
    assert [axis for axis, _ in _enabled_axes(config)] == ["frame"]
