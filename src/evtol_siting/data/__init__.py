"""数据获取与预处理模块。

* ``osm``       —— Overpass API 分块下载（建筑、POI、路网、用地、机场）
* ``heights``   —— 建筑高度多源融合与屋顶面积估计
* ``rasters``   —— 人口、收入代理、DEM/坡度栅格
* ``synthetic`` —— 合成数据（仅离线测试，禁止用于论文实证）
"""

from . import heights, osm, rasters, synthetic  # noqa: F401

__all__ = ["osm", "heights", "rasters", "synthetic"]
