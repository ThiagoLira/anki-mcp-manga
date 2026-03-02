from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .config import settings

if TYPE_CHECKING:
    from .anki_manager import AnkiManager

logger = logging.getLogger(__name__)

NO_CHANGES = 0
NORMAL_SYNC = 1
FULL_SYNC = 2
FULL_DOWNLOAD = 3
FULL_UPLOAD = 4


class SyncManager:
    def __init__(self, manager: AnkiManager):
        self._manager = manager

    def _do_sync(self, *, upload_on_full: bool) -> dict:
        """Run one sync cycle.

        Args:
            upload_on_full: When a full sync is required, True = upload
                local → server, False = download server → local.
        """
        col = self._manager.col

        auth = col.sync_login(
            username=settings.sync_user,
            password=settings.sync_password,
            endpoint=settings.sync_endpoint,
        )

        output = col.sync_collection(auth, sync_media=False)
        required = output.required

        result = {"collection_sync": "unknown", "media_sync": "pending"}

        if required == NO_CHANGES:
            result["collection_sync"] = "no_changes"
        elif required == NORMAL_SYNC:
            result["collection_sync"] = "synced"
        elif required in (FULL_UPLOAD, FULL_DOWNLOAD, FULL_SYNC):
            upload = upload_on_full
            label = "upload" if upload else "download"
            logger.info("Full sync required (code=%d), doing full %s", required, label)
            col.full_upload_or_download(
                auth=auth,
                server_usn=output.server_media_usn,
                upload=upload,
            )
            self._manager.reopen()
            result["collection_sync"] = f"full_{label}"

        # Re-fetch col after potential reopen (old reference is closed)
        col = self._manager.col

        auth = col.sync_login(
            username=settings.sync_user,
            password=settings.sync_password,
            endpoint=settings.sync_endpoint,
        )

        col.sync_media(auth)
        result["media_sync"] = "synced"

        return result

    def pull(self) -> dict:
        """Sync from server. On full-sync conflicts, always downloads
        (server is the source of truth)."""
        return self._do_sync(upload_on_full=False)

    def push(self) -> dict:
        """Sync to server. On full-sync conflicts, uploads
        (local has server state + new cards)."""
        return self._do_sync(upload_on_full=True)
