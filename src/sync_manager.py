from __future__ import annotations

from typing import TYPE_CHECKING

from .config import settings

if TYPE_CHECKING:
    from .anki_manager import AnkiManager


class SyncManager:
    def __init__(self, manager: AnkiManager):
        self._manager = manager

    def sync(self) -> dict:
        """Sync collection + media to the self-hosted sync server.

        Handles normal sync, full upload, and full download scenarios.
        """
        col = self._manager.col

        # Authenticate
        auth = col.sync_login(
            username=settings.sync_user,
            password=settings.sync_password,
            endpoint=settings.sync_endpoint,
        )

        # Attempt collection sync
        output = col.sync_collection(auth, sync_media=False)
        required = output.required

        result = {"collection_sync": "unknown", "media_sync": "pending"}

        NO_CHANGES = 0
        NORMAL_SYNC = 1
        FULL_UPLOAD = 4
        FULL_DOWNLOAD = 3

        if required == NO_CHANGES:
            result["collection_sync"] = "no_changes"
        elif required == NORMAL_SYNC:
            result["collection_sync"] = "synced"
        elif required == FULL_UPLOAD:
            col.full_upload_or_download(
                auth=auth,
                server_usn=output.server_media_usn,
                upload=True,
            )
            self._manager.reopen()
            result["collection_sync"] = "full_upload"
        elif required == FULL_DOWNLOAD:
            col.full_upload_or_download(
                auth=auth,
                server_usn=output.server_media_usn,
                upload=False,
            )
            self._manager.reopen()
            result["collection_sync"] = "full_download"
        else:
            # FULL_SYNC (2) — ambiguous, default to upload since we're the source
            col.full_upload_or_download(
                auth=auth,
                server_usn=output.server_media_usn,
                upload=True,
            )
            self._manager.reopen()
            result["collection_sync"] = "full_upload_forced"

        # Re-auth after potential reopen
        auth = col.sync_login(
            username=settings.sync_user,
            password=settings.sync_password,
            endpoint=settings.sync_endpoint,
        )

        # Sync media
        col.sync_media(auth)
        result["media_sync"] = "synced"

        return result
