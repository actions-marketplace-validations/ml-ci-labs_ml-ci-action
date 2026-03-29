#!/usr/bin/env python3
"""Switch the checked-out action manifest to the local Dockerfile runtime.

Used by PR self-tests so `uses: ./` exercises the in-repo Docker action
instead of the published GHCR image reference.
"""

from __future__ import annotations

import re
from pathlib import Path


ACTION_MANIFEST = Path("action.yml")
REMOTE_IMAGE_PATTERN = r"image: 'docker://[^']+'"
LOCAL_IMAGE_DECLARATION = "image: 'Dockerfile'"


def main() -> None:
    text = ACTION_MANIFEST.read_text(encoding="utf-8")
    updated, count = re.subn(REMOTE_IMAGE_PATTERN, LOCAL_IMAGE_DECLARATION, text, count=1)
    if count != 1:
        raise SystemExit("Failed to switch action runtime to Dockerfile for self-test")
    ACTION_MANIFEST.write_text(updated, encoding="utf-8")


if __name__ == "__main__":
    main()
