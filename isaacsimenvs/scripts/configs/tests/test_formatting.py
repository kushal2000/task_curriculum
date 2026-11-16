"""Configuration handed to loggers and policy loading contains only plain containers."""

from omegaconf import OmegaConf

from isaacsimenvs.scripts.configs.formatting import omegaconf_to_dict


def test_nested_lists_and_interpolations_become_plain_containers():
    cfg = OmegaConf.create({"width": 27, "layers": [{"size": "${width}"}, [1, 2]]})
    result = omegaconf_to_dict(cfg)
    assert result == {"width": 27, "layers": [{"size": 27}, [1, 2]]}
    assert type(result["layers"]) is list
    assert type(result["layers"][0]) is dict
    assert type(result["layers"][1]) is list
    result["layers"][0]["size"] = 3
    assert cfg.layers[0].size == 27
