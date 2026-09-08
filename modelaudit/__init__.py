# -*- coding: utf-8 -*-
"""Find the model defects that do not raise an exception.

See modelaudit/README.md for what each check does, what it cannot see, and the
measured base rates that put a hit in proportion.
"""

from .audit import audit, Report, SHARE, ZERO

__all__ = ['audit', 'Report', 'SHARE', 'ZERO']
