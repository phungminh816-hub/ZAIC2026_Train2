"""Các hàm dữ liệu được gom trong preprocess.py."""
from importlib import import_module

__all__ = ['ASRManifestDataset', 'SplitRoot', 'catalog_tools', 'filter_installed_tools', 'find_split', 'load_catalog', 'load_samples']


# Nạp muộn để python -m data.preprocess không chạy module hai lần.
def __getattr__(name: str):
    if name in __all__:
        value = getattr(import_module(".preprocess", __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
