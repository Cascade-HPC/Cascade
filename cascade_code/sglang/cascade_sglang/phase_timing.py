import json
import os
import threading
import time

_LOCK = threading.Lock()
_STATE = {
    "cache_seconds": 0.0,
    "cache_calls": 0,
    "store_seconds": 0.0,
    "store_calls": 0,
    "cache_hit_calls": 0,
    "lookup_seconds": 0.0,
    "lookup_calls": 0,
    "transfer_seconds": 0.0,
    "transfer_calls": 0,
    "publish_seconds": 0.0,
    "publish_calls": 0,
    "wire_seconds": 0.0,
    "wire_calls": 0,
    "load_seconds": 0.0,
    "load_calls": 0,
    "queue_seconds": 0.0,
    "prefill_seconds": 0.0,
    "timed_requests": 0,
}


def snapshot() -> dict:
    with _LOCK:
        out = dict(_STATE)
    try:
        from cascade_sglang import radix_cache as _rc
        counters = getattr(_rc, "COUNTERS", None)
        if counters is not None:
            with counters.lock:
                out["lookup_seconds"] += counters.lookup_seconds
                out["lookup_calls"] += counters.lookup_calls
                out["transfer_seconds"] += counters.get_seconds
                out["transfer_calls"] += counters.get_calls
                out["publish_async_seconds"] = counters.put_seconds
                out["publish_async_calls"] = counters.put_calls
                out["source"] = "cascade"
    except Exception:
        pass
    return out


def _add(field: str, value) -> None:
    with _LOCK:
        _STATE[field] += value


def install() -> bool:
    try:
        from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
    except Exception:
        return False

    seen = set()

    def wrap(cls):
        if cls in seen:
            return
        seen.add(cls)
        for name, field in (("match_prefix", "cache"),
                            ("cache_finished_req", "store")):
            fn = cls.__dict__.get(name)
            if fn is None or getattr(fn, "_phase_wrapped", False):
                continue

            def make(fn=fn, field=field, name=name):
                def wrapper(self, *a, **kw):
                    t0 = time.perf_counter()
                    try:
                        return fn(self, *a, **kw)
                    finally:
                        dt = time.perf_counter() - t0
                        _add(f"{field}_seconds", dt)
                        _add(f"{field}_calls", 1)
                wrapper._phase_wrapped = True
                wrapper.__name__ = name
                return wrapper

            setattr(cls, name, make())
        for sub in cls.__subclasses__():
            wrap(sub)

    wrap(BasePrefixCache)
    _wrap_storage()
    _wrap_request_timing()

    _orig_init_subclass = BasePrefixCache.__init_subclass__

    def _catch(cls, **kw):
        try:
            _orig_init_subclass.__func__(cls, **kw)
        except Exception:
            pass
        try:
            wrap(cls)
        except Exception:
            pass

    BasePrefixCache.__init_subclass__ = classmethod(_catch)
    return True


def _wrap_storage() -> None:
    try:
        from sglang.srt.mem_cache.hicache_storage import HiCacheStorage
    except Exception:
        return

    seen = set()

    def wrap(cls):
        if cls in seen:
            return
        seen.add(cls)
        for name, field in (("batch_exists", "lookup"),
                            ("batch_get_v1", "transfer"),
                            ("_get_batch_zero_copy_impl", "wire"),
                            ("batch_set_v1", "publish")):
            fn = cls.__dict__.get(name)
            if fn is None or getattr(fn, "_phase_wrapped", False):
                continue

            def make(fn=fn, field=field, name=name):
                def wrapper(self, *a, **kw):
                    t0 = time.perf_counter()
                    try:
                        return fn(self, *a, **kw)
                    finally:
                        _add(f"{field}_seconds", time.perf_counter() - t0)
                        _add(f"{field}_calls", 1)
                wrapper._phase_wrapped = True
                wrapper.__name__ = name
                return wrapper

            setattr(cls, name, make())
        for sub in cls.__subclasses__():
            wrap(sub)

    wrap(HiCacheStorage)

    _orig = HiCacheStorage.__init_subclass__

    def _catch(cls, **kw):
        try:
            _orig.__func__(cls, **kw)
        except Exception:
            pass
        try:
            wrap(cls)
        except Exception:
            pass

    HiCacheStorage.__init_subclass__ = classmethod(_catch)


def _wrap_request_timing() -> None:
    try:
        from sglang.srt.managers.schedule_batch import Req
    except Exception:
        return
    fn = Req.__dict__.get("log_time_stats")
    if fn is None or getattr(fn, "_phase_wrapped", False):
        return

    def wrapper(self, *a, **kw):
        try:
            ts = getattr(self, "time_stats", None)
            if ts is not None and not getattr(self, "has_log_time_stats", False):
                q = getattr(ts, "forward_entry_time", 0.0) - getattr(
                    ts, "wait_queue_entry_time", 0.0)
                pf = getattr(ts, "completion_time", 0.0) - getattr(
                    ts, "forward_entry_time", 0.0)
                if q >= 0 and pf >= 0:
                    _add("queue_seconds", q)
                    _add("prefill_seconds", pf)
                    _add("timed_requests", 1)
        except Exception:
            pass
        return fn(self, *a, **kw)

    wrapper._phase_wrapped = True
    wrapper.__name__ = "log_time_stats"
    Req.log_time_stats = wrapper


def write(path: str, extra: dict | None = None) -> None:
    data = snapshot()
    if extra:
        data.update(extra)
    with open(path, "w") as handle:
        json.dump(data, handle, indent=2)


def enabled() -> bool:
    return os.environ.get("CASCADE_PHASE_TIMING") == "1"


def start_reporter() -> None:
    directory = os.environ.get("CASCADE_PHASE_TIMING_DIR")
    if not directory:
        return
    try:
        os.makedirs(directory, exist_ok=True)
    except Exception:
        return
    name = "phase_{}_{}.json".format(
        os.environ.get("SLURM_PROCID", "0"), os.getpid())
    path = os.path.join(directory, name)

    def loop():
        while True:
            try:
                write(path, {
                    "pid": os.getpid(),
                    "procid": os.environ.get("SLURM_PROCID", "0"),
                    "tp_rank": os.environ.get("SGLANG_TP_RANK", ""),
                })
            except Exception:
                pass
            time.sleep(2.0)

    threading.Thread(target=loop, daemon=True, name="phase-timing").start()
