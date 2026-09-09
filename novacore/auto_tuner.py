"""
Auto-tuner: dynamically detect hardware and tune all knobs so the pipeline
runs at ~90% of available compute without manual config.

Detects:
  - CPU count + AVX2 support
  - Total + free RAM (Windows / Linux)
  - GPU + VRAM (CUDA / Apple MPS)
  - Disk free space

Tunes:
  - shard_rows_flush_cap     (parse batch size per shard)
  - vocab_build_cap          (tokens sampled for vocabulary)
  - pattern_sample_cap       (documents for pattern mining)
  - cooccurrence_window      (n-gram context size)
  - ngram_range              (min/max n-gram order)
  - text_stream_buffer       (file read buffer for TextStream)
  - num_threads              (BLAS / numpy threads)
  - use_gpu                  (whether to use GPU for encoder)
  - reservoir_size           (reservoir sampling cap)
  - memory_safety_margin     (free RAM to keep for OS)

Philosophy: never blow OOM. Always leave ~10% RAM headroom. Scale up
when free, scale down when memory tight.
"""
import os
import sys
import json
import time
import math
import multiprocessing
import subprocess


# ----------------------------------------------------------------------
# Hardware detection (cross-platform; no extra deps)
# ----------------------------------------------------------------------
def _cpu_count():
    try:
        return multiprocessing.cpu_count()
    except Exception:
        return os.cpu_count() or 1


def _has_avx2():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if "avx2" in line.lower():
                    return True
    except FileNotFoundError:
        pass
    # Windows: check via wmic / env var fallback
    if sys.platform.startswith("win"):
        try:
            out = subprocess.run(
                ["wmic", "cpu", "get", "Caption"],
                capture_output=True, text=True, timeout=5
            )
            # wmic is deprecated; if it fails, assume AVX2 (modern CPUs)
            return True
        except Exception:
            return True
    return False


def _total_ram_bytes():
    if sys.platform.startswith("win"):
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return int(stat.ullTotalPhys)
        except Exception:
            return 8 * 1024**3
    elif sys.platform == "darwin":
        try:
            out = subprocess.run(["sysctl", "-n", "hw.memsize"],
                                 capture_output=True, text=True, timeout=5)
            return int(out.stdout.strip())
        except Exception:
            return 8 * 1024**3
    else:
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) * 1024
        except Exception:
            pass
    return 8 * 1024**3


def _free_ram_bytes():
    if sys.platform.startswith("win"):
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return int(stat.ullAvailPhys)
        except Exception:
            return 2 * 1024**3
    else:
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) * 1024
        except Exception:
            pass
    return 2 * 1024**3


def _gpu_info():
    """Returns (available: bool, vram_bytes: int, name: str)."""
    # Try torch first (most reliable)
    try:
        import torch
        if torch.cuda.is_available():
            idx = 0
            free, total = torch.cuda.mem_get_info(idx)
            name = torch.cuda.get_device_name(idx)
            return True, total, name
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # Apple Silicon - shared memory
            return True, 0, "Apple MPS (unified memory)"
    except ImportError:
        pass
    # Try nvidia-smi as fallback
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,name", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if out.returncode == 0 and out.stdout.strip():
            line = out.stdout.strip().split("\n")[0]
            parts = [p.strip() for p in line.split(",")]
            vram_mb = float(parts[0])
            name = parts[1] if len(parts) > 1 else "NVIDIA GPU"
            return True, int(vram_mb * 1024 * 1024), name
    except Exception:
        pass
    return False, 0, ""


def _disk_free_bytes(path=None):
    try:
        if path is None:
            path = os.getcwd()
        if sys.platform.startswith("win"):
            import ctypes
            free_bytes = ctypes.c_ulonglong(0)
            ctypes.windll.kernel32.GetDiskFreeSpaceExW(
                ctypes.c_wchar_p(path), None, None, ctypes.byref(free_bytes)
            )
            return int(free_bytes.value)
        else:
            st = os.statvfs(path)
            return st.f_bavail * st.f_frsize
    except Exception:
        return 50 * 1024**3


# ----------------------------------------------------------------------
# Tuner
# ----------------------------------------------------------------------
class AutoTuner:
    """Detects hardware once, then returns tuned knobs on demand.

    All knobs are scaled so peak RAM stays at ~85% of total, leaving
    a 10% safety margin for the OS / other apps.
    """

    def __init__(self, safety=0.85):
        self.cpu_count = _cpu_count()
        self.avx2 = _has_avx2()
        self.total_ram = _total_ram_bytes()
        self.gpu_available, self.vram, self.gpu_name = _gpu_info()
        self.disk_free = _disk_free_bytes()
        self.safety = safety
        # max RAM we may use for buffers / vocab / patterns
        self.budget_ram = int(self.total_ram * safety)
        self._last_check = 0.0
        self._cached_free = self._free_now()

    # --- live monitoring (called between phases) ----------------------
    def _free_now(self):
        return _free_ram_bytes()

    def mem_pressure(self):
        """Return current memory pressure 0..1 (1 = at safety limit)."""
        used = self.total_ram - self._free_now()
        return min(1.0, used / max(1, self.budget_ram))

    def can_afford(self, bytes_needed):
        """True if allocating `bytes_needed` keeps us under safety."""
        return (self._free_now() - bytes_needed) > (self.total_ram * (1.0 - self.safety))

    def suggest_batch(self, base_rows):
        """Shrink `base_rows` if memory is tight; never blow OOM."""
        pressure = self.mem_pressure()
        if pressure > 0.9:
            return max(1000, base_rows // 4)
        if pressure > 0.75:
            return max(2000, base_rows // 2)
        if pressure < 0.5:
            return min(base_rows * 2, base_rows * 4)
        return base_rows

    # --- tuned knobs --------------------------------------------------
    def apply_threading(self):
        """Force BLAS / vectorized libs to use ~90% of cores for this process.
        Must ALSO be set before numpy is imported (see cli/main.py top)."""
        n = self.num_threads
        for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                  'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
            os.environ.setdefault(k, str(n))
        return n

    @property
    def num_threads(self):
        # leave 1-2 cores for OS / I/O
        return max(1, self.cpu_count - 2)

    @property
    def workers(self):
        """Worker processes for the parallel encode/extract pipeline.
        90% of cores, capped at 16 (I/O + RAM headroom); 1 when <=2 cores."""
        capped = max(1, min(16, int(self.cpu_count * 0.9)))
        return 1 if self.cpu_count < 4 else capped

    @property
    def shard_flush_cap(self):
        """Rows to hold in RAM before flushing to pool.jsonl per shard.

        Math: each pool row is ~9 KB (text only, no row duplication).
        So 50K rows ≈ 450 MB. We want at most 25% of budget.
        """
        target = self.budget_ram * 0.20
        per_row = 10 * 1024  # ~10 KB per pool row
        n = target // per_row
        # Cap between 5K (low RAM) and 100K (high RAM)
        if self.total_ram < 8 * 1024**3:    # < 8 GB
            return min(10_000, max(5_000, n))
        if self.total_ram < 16 * 1024**3:   # < 16 GB
            return min(50_000, max(10_000, n))
        if self.total_ram < 32 * 1024**3:   # < 32 GB
            return min(100_000, max(25_000, n))
        return min(200_000, max(50_000, n))

    @property
    def vocab_build_cap(self):
        """Documents to sample for vocabulary building.

        Vocabulary dict is roughly 1 MB per 50K unique tokens sampled
        from ~100K docs. Use ~15% of budget for vocab.
        """
        target = self.budget_ram * 0.15
        # ~1 MB per 50K sample docs (rough)
        n = (target // (1024 * 1024)) * 50_000
        if self.total_ram < 8 * 1024**3:
            return min(50_000, max(20_000, n))
        if self.total_ram < 16 * 1024**3:
            return min(200_000, max(50_000, n))
        if self.total_ram < 32 * 1024**3:
            return min(500_000, max(100_000, n))
        return min(1_000_000, max(200_000, n))

    @property
    def pattern_sample_cap(self):
        """Documents for pattern mining (co-occurrence pairs).

        Each doc contributes O(n-grams * window) pairs. Cap pairs at
        ~5% of budget to keep RAM under control.
        """
        target = self.budget_ram * 0.10
        # ~4 bytes per pair (int32 hash). 5% budget = 100K pairs/MB.
        n = (target // (4 * 1024 * 1024)) * 100_000
        if self.total_ram < 8 * 1024**3:
            return min(50_000, max(20_000, n))
        if self.total_ram < 16 * 1024**3:
            return min(200_000, max(50_000, n))
        return min(500_000, max(100_000, n))

    @property
    def cooccurrence_window(self):
        # Larger window needs more RAM for sliding pair buffers.
        if self.total_ram < 8 * 1024**3:
            return 3
        if self.total_ram < 16 * 1024**3:
            return 5
        return 7

    @property
    def ngram_range(self):
        # Higher n-grams explode pattern count. Cap by RAM.
        if self.total_ram < 8 * 1024**3:
            return (2, 3)
        if self.total_ram < 16 * 1024**3:
            return (2, 4)
        return (2, 5)

    @property
    def text_stream_buffer(self):
        # Larger buffer = more throughput but more RAM. ~2% of budget.
        b = int(self.budget_ram * 0.02)
        # 1 MB ... 64 MB
        return max(1 * 1024 * 1024, min(64 * 1024 * 1024, b))

    @property
    def reservoir_size(self):
        """Max rows kept in memory for reservoir sampling during streaming training."""
        # 15% of budget, ~5 KB per row (text content average)
        target = self.budget_ram * 0.15
        per_row = 5 * 1024
        n = target // per_row
        if self.total_ram < 8 * 1024**3:
            return min(50_000, max(10_000, n))
        if self.total_ram < 16 * 1024**3:
            return min(200_000, max(50_000, n))
        if self.total_ram < 32 * 1024**3:
            return min(500_000, max(100_000, n))
        return min(1_000_000, max(200_000, n))

    @property
    def use_gpu(self):
        return self.gpu_available

    def summary(self):
        gb = 1024**3
        return {
            "cpu_count": self.cpu_count,
            "avx2": self.avx2,
            "total_ram_gb": round(self.total_ram / gb, 1),
            "free_ram_gb": round(self._free_now() / gb, 1),
            "budget_ram_gb": round(self.budget_ram / gb, 1),
            "gpu_available": self.gpu_available,
            "gpu_name": self.gpu_name,
            "vram_gb": round(self.vram / gb, 1) if self.vram else 0,
            "disk_free_gb": round(self.disk_free / gb, 1),
            "num_threads": self.num_threads,
            "shard_flush_cap": self.shard_flush_cap,
            "vocab_build_cap": self.vocab_build_cap,
            "pattern_sample_cap": self.pattern_sample_cap,
            "cooccurrence_window": self.cooccurrence_window,
            "ngram_range": self.ngram_range,
            "text_stream_buffer": self.text_stream_buffer,
            "reservoir_size": self.reservoir_size,
        }


# ----------------------------------------------------------------------
# Module-level singleton (cached after first call)
# ----------------------------------------------------------------------
_tuner = None


def get_tuner():
    global _tuner
    if _tuner is None:
        _tuner = AutoTuner()
    return _tuner
