__all__ = ["CascadeHiCacheBackend"]


def __getattr__(name):
    if name == "CascadeHiCacheBackend":
        from cascade_sglang.backend import CascadeHiCacheBackend

        return CascadeHiCacheBackend
    raise AttributeError(name)
