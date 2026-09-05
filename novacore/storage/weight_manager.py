"""Storage: WeightManager - save/load/compress/query weights (NovaCore .ncw format)."""
import json
import os
import numpy as np


class WeightManager:
    """
    Manages the saved weights of a NovaCore model.

    NovaCore native format:
        - weights.ncw    : numpy compressed weight arrays (Nova Core Weights)
        - index.ncmeta   : metadata JSON (Nova Core Meta)
        - vocab.json     : vocabulary

    Backward compatible: purane weights.npz + index.json bhi load hote hain.
    """

    # NovaCore native filenames
    WEIGHTS_FILE = "weights.ncw"
    META_FILE = "index.ncmeta"
    VOCAB_FILE = "vocab.json"

    # Legacy filenames (backward compatible)
    LEGACY_WEIGHTS_FILE = "weights.npz"
    LEGACY_META_FILE = "index.json"

    # Temporary names used during the aborted first ncw attempt
    OLD_NCW_FILE = "model.ncw"
    OLD_NMETA_FILE = "model.nmeta"

    def __init__(self, model_dir):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

    @staticmethod
    def _get_arr(data, key, default=None):
        """Safely get array from npz dict."""
        try:
            arr = data[key]
            return arr
        except KeyError:
            return default

    def _resolve_files(self):
        """Return (weights_path, meta_path) preferring native, falling back to legacy."""
        native_w = os.path.join(self.model_dir, self.WEIGHTS_FILE)
        native_m = os.path.join(self.model_dir, self.META_FILE)
        legacy_w = os.path.join(self.model_dir, self.LEGACY_WEIGHTS_FILE)
        legacy_m = os.path.join(self.model_dir, self.LEGACY_META_FILE)

        if os.path.exists(native_w):
            return native_w, (native_m if os.path.exists(native_m) else legacy_m)
        if os.path.exists(legacy_w):
            return legacy_w, (legacy_m if os.path.exists(legacy_m) else native_m)
        return native_w, native_m

    def save(self, arrays, metadata):
        """Save weight arrays (weights.ncw) + metadata (index.ncmeta)."""
        # np.savez_compressed appends .npz; pass an open file object to keep
        # our exact .ncw filename.
        weights_path = os.path.join(self.model_dir, self.WEIGHTS_FILE)
        with open(weights_path, 'wb') as fh:
            np.savez_compressed(fh, **arrays)
        meta_path = os.path.join(self.model_dir, self.META_FILE)
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, default=str)

    def load(self):
        """Load weights + metadata (native .ncw or legacy .npz)."""
        w_path, m_path = self._resolve_files()
        if not os.path.exists(w_path):
            raise FileNotFoundError(f"Weights not found in: {self.model_dir}")
        data = np.load(w_path, allow_pickle=False)
        metadata = None
        if os.path.exists(m_path):
            with open(m_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        return data, metadata

    @classmethod
    def is_native(cls, model_dir):
        """Return True if the model uses the native .ncw format."""
        return os.path.exists(os.path.join(model_dir, cls.WEIGHTS_FILE))

    def load_array(self, key):
        """Load a specific key array without reading all."""
        data, _ = self.load()
        return self._get_arr(data, key)

    def info(self):
        """Get model info."""
        data, metadata = self.load()
        info = dict(metadata) if metadata else {}
        info['format'] = 'ncw' if WeightManager.is_native(self.model_dir) else 'legacy'
        info['keys'] = list(data.files)
        sizes = {}
        for key in data.files:
            arr = data[key]
            sizes[key] = {
                'shape': list(arr.shape),
                'dtype': str(arr.dtype),
                'bytes': int(arr.nbytes),
                'mb': round(arr.nbytes / (1024*1024), 3),
            }
        info['arrays'] = sizes
        total = sum(s['bytes'] for s in sizes.values())
        info['total_size_mb'] = round(total / (1024*1024), 3)
        return info

    def compress(self, ratio):
        """Compress weights by reducing float precision."""
        from ..config import get_default
        f16 = get_default('compress_f16_threshold')
        f32 = get_default('compress_f32_threshold')
        data, metadata = self.load()
        out = {}
        for key in data.files:
            arr = data[key]
            # Only compress floating-point arrays; keep int/uint/bytes intact
            if np.issubdtype(arr.dtype, np.floating):
                if ratio < f16:
                    out[key] = arr.astype(np.float16)
                elif ratio < f32:
                    out[key] = arr.astype(np.float32)
                else:
                    out[key] = arr
            else:
                out[key] = arr
        metadata = metadata or {}
        metadata['compressed'] = True
        metadata['compression_ratio'] = ratio
        self.save(out, metadata)

    def migrate(self):
        """Convert existing legacy (weights.npz/index.json) -> native (weights.ncw/index.ncmeta).
        Also repairs the aborted model.ncw naming if present."""
        # If aborted naming exists (model.ncw / model.nmeta), rename to native if no native present
        old_w = os.path.join(self.model_dir, self.OLD_NCW_FILE)
        old_m = os.path.join(self.model_dir, self.OLD_NMETA_FILE)
        native_w = os.path.join(self.model_dir, self.WEIGHTS_FILE)
        native_m = os.path.join(self.model_dir, self.META_FILE)
        migrated_old = False
        if os.path.exists(old_w):
            os.rename(old_w, native_w)
            migrated_old = True
        if os.path.exists(old_m):
            os.rename(old_m, native_m)
        # drop any old leftover .npz suffix file if present
        leftover = old_w + ".npz"
        if os.path.exists(leftover):
            os.remove(leftover)

        # Legacy -> native conversion
        legacy_w = os.path.join(self.model_dir, self.LEGACY_WEIGHTS_FILE)
        legacy_m = os.path.join(self.model_dir, self.LEGACY_META_FILE)
        if os.path.exists(legacy_w) and not os.path.exists(native_w):
            # Use a scoped load so the file handle is released before removal
            loaded = np.load(legacy_w, allow_pickle=False)
            arrays = dict(loaded)
            loaded.close()
            metadata = None
            if os.path.exists(legacy_m):
                with open(legacy_m, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            self.save(arrays, metadata or {})
            # remove legacy files after successful conversion
            if os.path.exists(native_w):
                os.remove(legacy_w)
                if os.path.exists(legacy_m):
                    os.remove(legacy_m)
            return True
        return migrated_old

    @staticmethod
    def list_models(base_dir):
        """List all saved models in a base directory (native .ncw or legacy)."""
        models = []
        if not os.path.exists(base_dir):
            return models
        for entry in os.listdir(base_dir):
            full = os.path.join(base_dir, entry)
            if not os.path.isdir(full):
                continue
            has_model = (
                os.path.exists(os.path.join(full, WeightManager.WEIGHTS_FILE))
                or os.path.exists(os.path.join(full, WeightManager.LEGACY_WEIGHTS_FILE))
                or os.path.exists(os.path.join(full, WeightManager.OLD_NCW_FILE))
            )
            if has_model:
                models.append(entry)
        return models
