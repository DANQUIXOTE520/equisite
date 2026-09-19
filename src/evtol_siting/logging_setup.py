"""日志配置。

为什么需要专门的模块
--------------------
在中文 Windows 上，控制台默认编码是 GBK（cp936）。本项目的中文日志里
含有 ``²``（如 "个/km²"）、``≥``、``→`` 等 GBK 无法表示的字符，直接
``print``/``logging`` 会抛 ``UnicodeEncodeError``。

该异常发生在日志写入阶段，**不会让程序崩溃**，但会让那条日志完全丢失，
并在 stderr 打出难懂的 traceback——在长时间的批处理中，这会掩盖真正的
问题。因此这里显式把输出流包成 UTF-8，并对无法编码的字符做替换而非抛错。
"""

from __future__ import annotations

import io
import logging
import sys
from pathlib import Path


def _utf8_stream(stream):
    """把文本流重新包成 UTF-8，无法编码的字符替换而非抛异常。

    已经是 UTF-8 的流原样返回，避免重复包装。
    """
    if stream is None:
        return None
    enc = (getattr(stream, "encoding", None) or "").lower().replace("-", "")
    if enc in ("utf8", "utf8mb4"):
        return stream
    try:
        buf = getattr(stream, "buffer", None)
        if buf is None:
            return stream
        return io.TextIOWrapper(
            buf, encoding="utf-8", errors="replace", line_buffering=True
        )
    except Exception:
        return stream


def setup_logging(
    verbose: bool = False,
    log_file: str | Path | None = None,
    quiet_libraries: bool = True,
) -> None:
    """配置根日志器：控制台（UTF-8 安全）+ 可选文件。

    Parameters
    ----------
    verbose
        ``True`` 时输出 DEBUG 级日志。
    log_file
        同时写入该文件（UTF-8）。论文要求保留每次实验的完整运行日志，
        以便复核参数与随机种子。
    quiet_libraries
        抑制 matplotlib / fiona / geopandas 等第三方库的 INFO 噪声。
    """
    level = logging.DEBUG if verbose else logging.INFO

    # 关键：先把标准输出转成 UTF-8，再挂 handler
    sys.stdout = _utf8_stream(sys.stdout)
    sys.stderr = _utf8_stream(sys.stderr)

    handlers: list[logging.Handler] = []
    if sys.stdout is not None:
        handlers.append(logging.StreamHandler(sys.stdout))
    if log_file:
        p = Path(log_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(p, encoding="utf-8", errors="replace"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)-26s %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )

    if quiet_libraries:
        for noisy in ("matplotlib", "fiona", "geopandas", "rasterio",
                      "pyproj", "urllib3", "shapely", "PIL"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
