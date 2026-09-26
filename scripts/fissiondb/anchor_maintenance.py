"""Bounded-frequency journal maintenance for an open AnchorIndex."""
import math
import operator
import threading
import weakref


class AutoCompactor:
    def __init__(self, index, growth_bytes, interval):
        growth_bytes = operator.index(growth_bytes)
        if not index.live or growth_bytes < 1 or not math.isfinite(interval) or interval <= 0:
            raise ValueError('Automatic compaction requires live mode and positive thresholds')
        self._index = weakref.ref(index)
        self._baseline = 0
        self.growth_bytes, self.interval = growth_bytes, interval
        self.last_error, self.last_result, self.runs = None, None, 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name='anchor-compactor', daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.wait(self.interval):
            index = self._index()
            if index is None:
                return
            try:
                size = index.maintenance_bytes
                if size < self._baseline:
                    self._baseline = size
                if size - self._baseline >= self.growth_bytes:
                    self.last_result = index.compact()
                    self._baseline = index.maintenance_bytes
                    self.runs += 1
            except Exception as exc:
                self.last_error = str(exc)
                return  # Latch failures; do not repeatedly rewrite a failing disk.
            finally:
                del index

    def close(self):
        self._stop.set()
        if threading.current_thread() is not self._thread:
            self._thread.join()


class AutoPacker(AutoCompactor):
    """Schedule live packing or a fission checkpoint at a byte threshold.

    Adaptive checkpoints acquire the live write lock and can delay queries.
    """
    def __init__(self, index, growth_bytes, interval):
        if index.residual_dir is None:
            raise ValueError('Automatic packing requires residual mode')
        super().__init__(index, growth_bytes, interval)

    def _run(self):
        while not self._stop.wait(self.interval):
            index = self._index()
            if index is None:
                return
            try:
                if index.unpacked_bytes >= self.growth_bytes:
                    index.pack_live()
                    self.runs += 1
                    self.last_result = {'unpacked_bytes': index.unpacked_bytes}
            except Exception as exc:
                self.last_error = str(exc)
                return
            finally:
                del index
