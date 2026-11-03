"""可信任务资产根检查；不允许资源写入经 symlink/junction 跳出的目录。"""
import os
from pathlib import Path
import stat

MAX_ASSET_BYTES = 16 * 1024 * 1024
MAX_JOB_BYTES = 128 * 1024 * 1024
MAX_ASSETS = 500


def inventory(root):
    count, total, directories = 0, 0, 0
    for current, dirs, files in os.walk(root, followlinks=False):
        directories += len(dirs)
        if directories > MAX_ASSETS * 2:
            raise ValueError("ASSET_OUTPUT_BUDGET: too many asset directories")
        for name in dirs + files:
            item = Path(current) / name
            info = item.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                raise ValueError("ASSET_OUTPUT_INVALID: linked asset entry")
        for name in files:
            size = (Path(current) / name).stat().st_size
            count += 1
            total += size
            if count > MAX_ASSETS or total > MAX_JOB_BYTES or size > MAX_ASSET_BYTES:
                raise ValueError("ASSET_OUTPUT_BUDGET: existing job assets exceed budget")
    return total, count


class LimitedWriter:
    """禁止编码器通过 fileno 绕开 Python 写入上限；超限前停止写入。"""
    def __init__(self, stream, limit):
        self.stream, self.limit = stream, limit
    def write(self, data):
        if self.stream.tell() + len(data) > self.limit:
            raise ValueError("ASSET_OUTPUT_BUDGET: encoded image exceeds remaining job or 16 MiB limit")
        return self.stream.write(data)
    def fileno(self):
        raise OSError("bounded image output has no direct file descriptor")
    def __getattr__(self, name):
        return getattr(self.stream, name)


def output_root(value):
    root = Path(value)
    if not root.is_absolute() or ".." in root.parts:
        raise ValueError("ASSET_OUTPUT_INVALID: assetOutputRoot must be an absolute owned directory")
    # 检查整个既有祖先链，包括目标自身；不能只比较字符串前缀。
    for current in reversed((root, *root.parents)):
        try:
            info = current.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("ASSET_OUTPUT_INVALID: reparse point in asset output directory")
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError("ASSET_OUTPUT_INVALID: asset output ancestor is not a directory")
    if not root.parent.is_dir():
        raise ValueError("ASSET_OUTPUT_INVALID: job directory must already exist")
    root.mkdir(exist_ok=True)
    if os.path.normcase(str(root.absolute())) != os.path.normcase(str(root.resolve(strict=True))):
        raise ValueError("ASSET_OUTPUT_INVALID: asset output escapes its owned directory")
    return root
