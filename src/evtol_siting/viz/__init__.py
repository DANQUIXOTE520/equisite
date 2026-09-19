"""制图模块（出版级图件输出）。

* ``figures`` —— 统计图：高度融合、人群分类、帕累托前沿、收敛曲线、
  敏感性分析、方案对比、选择模型机制
* ``maps``    —— 空间图（在 figures 中以 ``fig_study_area`` /
  ``fig_accessibility`` 实现，故本模块暂不单独提供）

图件生成入口见 ``scripts/09_figures.py``，输出到项目根的 ``论文稿件/图件/``。
"""

from . import figures  # noqa: F401

__all__ = ["figures"]
