"""Product identity.

CLAUDE.md line 6: the codename lives here as a single constant and renaming must touch
nothing else. Every user-visible string that names the product derives from APP_NAME.
The console script names in pyproject.toml are deliberately excluded: entry point names
are packaging metadata, not UI, and renaming them would break installed shells.
"""

from __future__ import annotations

APP_NAME = "DRIVYX"

#: Window titles, banners, and report headers.
APP_TITLE = APP_NAME

#: Qt application/organisation identity, used for QSettings paths.
ORG_NAME = APP_NAME
