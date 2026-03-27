#!/usr/bin/env python3
"""Marlin MultiQuant adapter tests."""

import pytest


class TestMarlinAdapter:
    def test_config_creation(self):
        from vllm.multiquant.marlin.config import MarlinMultiQuantConfig
        cfg = MarlinMultiQuantConfig(bits=4, group_size=128)
        assert cfg.bits == 4
        assert cfg.get_name() == "marlin_mq"

    def test_config_from_config(self):
        from vllm.multiquant.marlin.config import MarlinMultiQuantConfig
        cfg = MarlinMultiQuantConfig.from_config({"bits": 4})
        assert cfg.bits == 4

    def test_min_capability(self):
        from vllm.multiquant.marlin.config import MarlinMultiQuantConfig
        assert MarlinMultiQuantConfig.get_min_capability() == 80


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
