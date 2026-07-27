from __future__ import annotations

import logging

from cywl_oopz.core.health import HealthRegistry, HealthState
from cywl_oopz.core.observability import exception_kind, opaque_ref


def test_opaque_ref_is_stable_and_does_not_render_source_values() -> None:
    reference = opaque_ref("area-secret", "channel-secret", "person-secret")

    assert reference == opaque_ref("area-secret", "channel-secret", "person-secret")
    assert len(reference) == 12
    assert "secret" not in reference
    assert exception_kind(ValueError("hidden")) == "ValueError"


def test_health_registry_logs_only_health_transitions(caplog) -> None:
    registry = HealthRegistry()
    caplog.set_level(logging.INFO, logger="cywl_oopz.core.health")

    registry.mark("database", HealthState.HEALTHY, "connected")
    registry.mark("database", HealthState.HEALTHY, "connected")
    registry.mark("database", HealthState.DEGRADED, "query failed")

    assert [record.message for record in caplog.records] == [
        "Component health changed: component=database state=healthy detail=connected",
        "Component health changed: component=database state=degraded detail=query failed",
    ]
