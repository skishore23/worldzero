"""Bundled exact built-in law-family implementations and legacy mapping."""

from __future__ import annotations

from ..registry import RegisteredFamily, fingerprint_family
from .catalysis import CatalysisFamily
from .delayed_transformation import DelayedTransformationFamily
from .inhibition import InhibitionFamily
from .null import NullFamily


_LEGACY_FAMILIES = {
    "catalysis": CatalysisFamily,
    "null": NullFamily,
}


def builtin_families() -> tuple[CatalysisFamily | DelayedTransformationFamily | InhibitionFamily | NullFamily, ...]:
    """Return fresh built-ins in exact family-ID order."""

    return (CatalysisFamily(), DelayedTransformationFamily(), InhibitionFamily(), NullFamily())


def legacy_registered_family(name: str) -> RegisteredFamily:
    """Resolve one fixed legacy short name without package metadata lookup."""

    try:
        family = _LEGACY_FAMILIES[name]()
    except KeyError as exc:  # Law validates this before the mapping boundary.
        raise ValueError(f"Unknown legacy law family: {name}") from exc
    return RegisteredFamily(
        family=family,
        origin="legacy-adapter",
        official=True,
        fingerprint=fingerprint_family(family),
    )


__all__ = [
    "CatalysisFamily",
    "DelayedTransformationFamily",
    "InhibitionFamily",
    "NullFamily",
    "builtin_families",
    "legacy_registered_family",
]
