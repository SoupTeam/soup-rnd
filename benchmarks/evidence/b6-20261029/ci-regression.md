# TOCTOU capture regression

The original Windows simulation replaced process-wide os.name/os.fstat/os.lstat.
A regression that stats an unrelated real descriptor while the swap mock is
active reproduced AttributeError: MockSwappedFstat has no attribute st_size.
The helper's TOCTOU rejection still occurred before the capture failure.

The corrected fixtures replace soup_cli.utils.paths.os with a local namespace.
All 15 tests in test_issue820_windows_symlink_open_nofollow.py pass on the local
Windows/Python 3.10 host, including the active-mock descriptor/capture regression.
No production symlink or TOCTOU check is removed or weakened.
