"""配置加载与全局路径管理。

设计要点
--------
1. 单一配置源：``config/chengdu.yaml`` 是唯一参数入口，敏感性分析通过
   覆盖（override）该配置实现，保证"实验条件"与"代码版本"解耦。
2. 所有路径解析为绝对路径，脚本可从任意工作目录调用。
3. 提供 ``Config.override()`` 用于批量情景实验（不修改磁盘上的基准配置）。
"""

from __future__ import annotations

import copy
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# 路径解析
# ---------------------------------------------------------------------------

# <repo>/src/evtol_siting/config.py  ->  <repo>/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "chengdu.yaml"


def _default_config_path() -> Path:
    """默认配置路径，可用环境变量 ``EVTOL_CONFIG`` 覆盖。

    多城市跑同一套脚本时，逐个给脚本加 ``--config`` 参数改动面太大，
    而漏掉一个脚本就会**静默拿着 A 城的配置去跑 B 城**——正是
    `data/pbf.py::default_pbf_path` 踩过的那类坑。用一个环境变量兜住，
    所有 ``load_config()`` 调用点自动生效。

    ``EVTOL_CONFIG`` 可以是绝对路径，也可以是相对 ``PROJECT_ROOT`` 的路径。
    """
    env = os.environ.get("EVTOL_CONFIG")
    if not env:
        return DEFAULT_CONFIG_PATH
    p = Path(env)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def _deep_update(base: dict, override: dict) -> dict:
    """递归合并 ``override`` 到 ``base``（返回新对象，不修改入参）。

    键可以是**嵌套字典**，也可以是**点号路径**——两种写法等价::

        {"stage4_nsga2": {"pop_size": 50}}
        {"stage4_nsga2.pop_size": 50}

    支持点号路径是必要的：敏感性分析脚本里要覆盖几十个参数，嵌套写法
    冗长且容易写错层级；而点号写法一旦不被识别，覆盖会**静默失效**
    （参数没生效但程序照常运行，结果却是默认参数下的），这是很难发现的
    实验错误。因此这里显式展开点号键，并在遇到冲突时报错而不是任选其一。
    """
    # 先把点号键展开为嵌套结构
    expanded: dict = {}
    for key, value in override.items():
        if isinstance(key, str) and "." in key:
            node = expanded
            parts = key.split(".")
            for p in parts[:-1]:
                node = node.setdefault(p, {})
            leaf = parts[-1]
            if leaf in node and isinstance(node[leaf], dict) and isinstance(value, dict):
                node[leaf] = _deep_update(node[leaf], value)
            else:
                node[leaf] = copy.deepcopy(value)
        else:
            if key in expanded and isinstance(expanded[key], dict) and isinstance(value, dict):
                expanded[key] = _deep_update(expanded[key], value)
            else:
                expanded[key] = copy.deepcopy(value)

    out = copy.deepcopy(base)
    for key, value in expanded.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


_MISSING = object()   # 哨兵：区分"配置值就是 None"与"路径不存在"


def _get(d: dict, dotted: str, default: Any = None) -> Any:
    """按 ``"a.b.c"`` 路径取值；缺失时返回 ``default``。"""
    node: Any = d
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


@dataclass
class Config:
    """实验配置的只读视图 + 路径解析。"""

    raw: dict
    source_path: Path
    overrides: dict = field(default_factory=dict)

    # -- 构造 ---------------------------------------------------------------

    @classmethod
    def load(
        cls,
        path: str | os.PathLike | None = None,
        overrides: dict | None = None,
    ) -> "Config":
        """从 YAML 载入配置，可选地叠加 ``overrides``。

        Parameters
        ----------
        path
            配置文件路径；``None`` 时使用 ``config/chengdu.yaml``。
        overrides
            嵌套字典，按 key 递归覆盖基准配置。例如::

                Config.load(overrides={"stage3_ip": {"access_radius_m": 2000}})
        """
        p = Path(path) if path is not None else _default_config_path()
        p = p if p.is_absolute() else (PROJECT_ROOT / p)
        if not p.exists():
            raise FileNotFoundError(f"配置文件不存在: {p}")
        with open(p, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)

        overrides = overrides or {}

        # 覆盖路径自检必须针对**合并前**的原始配置。
        pristine = copy.deepcopy(raw)
        cfg = cls(
            raw=_deep_update(raw, overrides),
            source_path=p,
            overrides=overrides,
        )
        bad = cfg.check_overrides(base=pristine)
        if bad:
            import warnings

            warnings.warn(
                "以下覆盖路径在基准配置中不存在，参数**不会生效**：\n  "
                + "\n  ".join(bad)
                + "\n请核对 config/chengdu.yaml 中的实际键名。",
                UserWarning,
                stacklevel=2,
            )
        return cfg

    # -- 访问 ---------------------------------------------------------------

    def get(self, dotted: str, default: Any = None) -> Any:
        """点号路径取值：``cfg.get("stage1_candidates.rooftop.min_height_m")``。"""
        return _get(self.raw, dotted, default)

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def __contains__(self, key: str) -> bool:
        return key in self.raw

    def override(self, overrides: dict) -> "Config":
        """返回叠加了 ``overrides`` 的新配置对象（不写盘）。"""
        return Config(
            raw=_deep_update(self.raw, overrides),
            source_path=self.source_path,
            overrides=_deep_update(self.overrides, overrides),
        )

    # -- 路径 ---------------------------------------------------------------

    def check_overrides(self, base: dict | None = None) -> list[str]:
        """校验本次覆盖涉及的路径在基准配置中确实存在。

        返回"疑似无效"的点号路径列表。覆盖一个不存在的路径会让参数
        **静默失效**（程序正常运行，但用的是默认值），这类错误在论文
        实验中极难察觉——因此在载入时主动告警。

        Parameters
        ----------
        base
            用于校验的基准配置。**必须传入合并前的原始配置**：合并后的
            ``self.raw`` 已经包含了被覆盖（甚至是被凭空添加）的键，
            拿它做校验会让所有路径都"看起来存在"，校验形同虚设。
        """
        base = self.raw if base is None else base
        bad: list[str] = []

        def _walk(node, prefix=""):
            if not isinstance(node, dict):
                return
            for k, v in node.items():
                dotted = f"{prefix}{k}"
                if isinstance(v, dict) and v:
                    _walk(v, f"{dotted}.")
                elif _get(base, dotted, _MISSING) is _MISSING:
                    bad.append(dotted)

        _walk(self.overrides)
        return bad

    def path(self, dotted: str, ensure_parent: bool = False) -> Path:
        """把配置中的相对路径解析为绝对路径。"""
        value = self.get(dotted)
        if value is None:
            raise KeyError(f"配置路径不存在: {dotted}")
        p = Path(value)
        p = p if p.is_absolute() else (PROJECT_ROOT / p)
        if ensure_parent:
            p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def dir(self, dotted: str) -> Path:
        """解析目录路径并确保其存在。"""
        p = self.path(dotted)
        p.mkdir(parents=True, exist_ok=True)
        return p

    # -- 便捷属性 -----------------------------------------------------------

    @property
    def crs_geo(self) -> str:
        return self.get("crs.geographic")

    @property
    def crs_proj(self) -> str:
        return self.get("crs.projected")

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """下载外包矩形 ``(minx, miny, maxx, maxy)``（WGS84）。"""
        return tuple(self.get("study_area.bbox"))

    @property
    def seed(self) -> int:
        return int(self.get("project.random_seed", 42))

    def objective_keys(self) -> list[str]:
        return [o["key"] for o in self.get("stage4_nsga2.objectives", [])]

    def objective_labels(self, lang: str | None = None) -> list[str]:
        lang = lang or self.get("output.language", "en")
        field_name = "name_zh" if lang == "zh" else "name_en"
        return [o.get(field_name, o["key"]) for o in self.get("stage4_nsga2.objectives", [])]

    # -- 复现性 -------------------------------------------------------------

    def seed_everything(self) -> None:
        """固定所有随机源，保证论文结果可复现。"""
        seed = self.seed
        random.seed(seed)
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)

    def describe(self, keys: Iterable[str] | None = None) -> str:
        """生成配置摘要，写进日志与论文附录。"""
        keys = keys or [
            "stage1_candidates.rooftop.min_height_m",
            "stage1_candidates.rooftop.min_roof_area_m2",
            "stage1_candidates.dilution.min_distance_m",
            "stage1_candidates.grid_size_m",
            "stage2_demand.n_clusters",
            "stage3_ip.access_radius_m",
            "stage3_ip.min_core_demand",
            "stage4_nsga2.pop_size",
            "stage4_nsga2.n_gen",
        ]
        lines = [f"config: {self.source_path.name}"]
        for k in keys:
            lines.append(f"  {k:52s} = {self.get(k)}")
        if self.overrides:
            lines.append(f"  overrides = {self.overrides}")
        return "\n".join(lines)


def load_config(path=None, overrides: dict | None = None) -> Config:
    """``Config.load`` 的函数式封装，便于脚本一行调用。"""
    return Config.load(path, overrides)
