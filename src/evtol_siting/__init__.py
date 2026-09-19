"""基于 GIS 与多目标优化的 eVTOL 起降场选址研究。

四阶段集成方法::

    阶段一  GIS 候选筛选       stage1_candidates
    阶段二  K-means 人群分类    stage2_demand
    阶段三  0-1 整数规划选址    stage3_ip
    阶段四  NSGA-II 多目标择优  stage4_nsga2

辅助模块：``geo``（几何原语）、``metrics``（评价指标）、
``baselines``（单目标基线）、``nsga2_core``（算法实现）、``viz``（制图）。
"""

__version__ = "0.1.0"

from .config import Config, PROJECT_ROOT, load_config

__all__ = ["Config", "PROJECT_ROOT", "load_config", "__version__"]
