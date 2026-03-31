"""
FileService — wraps IFileService via the C bridge.
"""


class FileService:
    """Manages robot-side program files and directories."""

    def __init__(self, lib, ctx):
        self._lib = lib
        self._ctx = ctx

    def mkdir(self, path: str, recursive: bool = False) -> bool:
        return bool(self._lib.crp_file_mkdir(self._ctx,
                                              path.encode(), recursive))

    def rmdir(self, path: str, recursive: bool = False) -> bool:
        return bool(self._lib.crp_file_rmdir(self._ctx,
                                              path.encode(), recursive))

    def rename(self, src: str, dst: str) -> bool:
        return bool(self._lib.crp_file_rename(self._ctx,
                                               src.encode(), dst.encode()))

    def copy(self, src: str, dst: str) -> bool:
        return bool(self._lib.crp_file_copy(self._ctx,
                                             src.encode(), dst.encode()))

    def remove(self, path: str) -> bool:
        return bool(self._lib.crp_file_remove(self._ctx, path.encode()))

    def exists(self, path: str) -> bool:
        return bool(self._lib.crp_file_exists(self._ctx, path.encode()))

    def upload(self, local_path: str, remote_path: str) -> bool:
        return bool(self._lib.crp_file_upload(self._ctx,
                                               local_path.encode(),
                                               remote_path.encode()))

    def download(self, remote_path: str, local_path: str) -> bool:
        return bool(self._lib.crp_file_download(self._ctx,
                                                 remote_path.encode(),
                                                 local_path.encode()))
