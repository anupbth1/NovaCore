"""Dataset loader - support multiple formats."""
import csv
import json
import os


class DatasetLoader:
    """
    Loads datasets from various formats.
    Returns (texts, meta) where texts is a list of strings.
    """

    SUPPORTED = ['csv', 'json', 'txt', 'tsv']

    @staticmethod
    def load(path, format=None, text_column=None):
        """
        Load dataset.

        Args:
            path: Dataset file path
            format: auto/csv/json/txt/tsv (None=auto-detect)
            text_column: Column to use as text (CSV/JSON)

        Returns:
            (raw_texts, meta) tuple
        """
        fmt = format or DatasetLoader._detect_format(path)
        if fmt not in DatasetLoader.SUPPORTED:
            raise ValueError(f"Unsupported format: {fmt}")

        if fmt == 'txt':
            return DatasetLoader._load_txt(path), {'format': 'txt'}
        elif fmt == 'csv':
            return DatasetLoader._load_csv(path, text_column, ','), {'format': 'csv'}
        elif fmt == 'tsv':
            return DatasetLoader._load_csv(path, text_column, '\t'), {'format': 'tsv'}
        elif fmt == 'json':
            return DatasetLoader._load_json(path, text_column), {'format': 'json'}
        raise ValueError(f"Unknown format: {fmt}")

    @staticmethod
    def _detect_format(path):
        ext = os.path.splitext(path)[1].lower().lstrip('.')
        if ext in DatasetLoader.SUPPORTED:
            return ext
        raise ValueError(f"Cannot detect format for: {path}")

    @staticmethod
    def _load_txt(path):
        """Load text file, one document per line."""
        texts = []
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if line:
                    texts.append(line)
        return texts

    @staticmethod
    def _load_csv(path, text_column, delimiter=','):
        """Load CSV/TSV file."""
        texts = []
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            if not reader.fieldnames:
                raise ValueError("Empty CSV/TSV file")
            column = text_column or reader.fieldnames[0]
            if column not in reader.fieldnames:
                raise ValueError(f"Column '{column}' not found. Available: {reader.fieldnames}")
            for row in reader:
                val = row.get(column, '')
                if val and val.strip():
                    texts.append(val.strip())
        return texts

    @staticmethod
    def _auto_preferred(keys):
        """Pick a preferred text column from available keys (config-driven order)."""
        try:
            from ..config import get_hf
            preferred = get_hf('auto_columns', None)
        except Exception:
            preferred = None
        if preferred:
            for name in preferred:
                if name in keys:
                    return name
        return keys[0] if keys else None

    @staticmethod
    def _load_json(path, text_column):
        """Load JSON file (array or object with list)."""
        texts = []
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Handle: [ {"text": "..."}, ... ] or [ "str1", "str2" ] or {"data": [...]}
        if isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    if item.strip():
                        texts.append(item.strip())
                elif isinstance(item, dict):
                    if not item:
                        continue
                    column = text_column or DatasetLoader._auto_preferred(list(item.keys()))
                    if column and column in item:
                        val = str(item[column])
                        if val.strip():
                            texts.append(val.strip())
        elif isinstance(data, dict):
            for key in data:
                if isinstance(data[key], list):
                    for item in data[key]:
                        if isinstance(item, str) and item.strip():
                            texts.append(item.strip())
                        elif isinstance(item, dict):
                            if not item:
                                continue
                            column = text_column or DatasetLoader._auto_preferred(list(item.keys()))
                            if column and column in item and str(item[column]).strip():
                                texts.append(str(item[column]).strip())
        return texts
