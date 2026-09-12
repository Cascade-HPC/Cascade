from __future__ import annotations

import os
import sys
import types

_MODULE = "sglang.srt.mem_cache.storage.lmcache.lmc_radix_cache"
_installed = [False]


def install_device_cache() -> bool:
    if _installed[0]:
        return True
    if os.environ.get("CASCADE_SGLANG_DEVICE", "") not in ("1", "true", "yes"):
        return False
    if _MODULE in sys.modules:
        existing = getattr(sys.modules[_MODULE], "LMCRadixCache", None)
        if existing is not None and existing.__name__ != "CascadeRadixCache":
            raise RuntimeError(
                "LMCache's radix cache module was imported before the Cascade "
                "device hook could replace it"
            )
        return True

    shim = types.ModuleType(_MODULE)

    class _Lazy:

        def __call__(self, *args, **kwargs):
            from cascade_sglang.radix_cache import CascadeRadixCache

            return CascadeRadixCache(*args, **kwargs)

        __name__ = "CascadeRadixCache"

    shim.LMCRadixCache = _Lazy()
    shim.__cascade_shim__ = True
    sys.modules[_MODULE] = shim
    _installed[0] = True
    return True
