"""Domain-specific fail-closed errors."""


class AeroCityError(RuntimeError):
    """Base error for user-facing build and validation failures."""


class AssetRegistryError(AeroCityError):
    """An asset registry or one of its files failed validation."""


class GenerationRejected(AeroCityError):
    """A deterministic candidate failed an admission rule."""


class ValidationError(AeroCityError):
    """A generated release failed integrity or scientific validation."""


class HostGuardError(AeroCityError):
    """The execution host is unsafe or failed independently of benchmark science."""
