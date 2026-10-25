"""Keep the experiment self contained on Windows with an older system VC runtime."""
import ctypes
import os
import sys
from pathlib import Path

_dll_handles = []
if os.name == "nt":
    _runtime_dir = Path(sys.executable).parent
    if (_runtime_dir / "msvcp140.dll").exists():
        _dll_handles.append(os.add_dll_directory(str(_runtime_dir)))
        _dll_handles.append(ctypes.WinDLL(str(_runtime_dir / "msvcp140.dll")))
