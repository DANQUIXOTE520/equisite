"""MILP 求解状态的**可信判据**（pulp + CBC/HiGHS 两条通道）。

为什么必须单独成模块
--------------------
审计发现：本判据最初只在 ``baselines.py`` 里实现，而 ``stage3_ip`` /
``schemes`` / ``scenarios`` 三处仍在用 ``pulp.LpStatus[...] == "Optimal"``
这一**不可靠**的判据，于是论文里"已证明最优"的说法只在一处站得住。
判据必须**只有一份实现**，否则修了一处、漏了三处，而且漏掉的地方
恰好是正文引用的那几处。

两个求解器各自的坑
------------------
**CBC**：``readsol`` 里解文件首行第一个词若是 ``Stopped``（时限用尽或达到
间隙），只要第 5 个词是 ``objective``，状态就被改写成 ``LpStatusOptimal``
（``pulp/apis/coin_api.py::get_status``）。所以仅凭 ``LpStatus`` 无法区分
"证明最优"与"截断的最好解"——这正是早期 p-中位 32 % 覆盖率被当成最优解的来路。
CBC 下可靠判据是 ``sol_status == LpSolutionOptimal``。

**HiGHS**（更隐蔽）：HiGHS 达到 ``mip_rel_gap`` 容差而停时，模型状态仍是
``kOptimal``，被 pulp 映射成 ``LpStatusOptimal`` + ``LpSolutionOptimal``
（``pulp/apis/highs_api.py:447-450``）。**于是 ``sol_status`` 在 HiGHS 下彻底
失效**：实测把 gap 容差设成 5 %、实际 gap 0.41 % 停下时，``sol_status`` 依然
报 ``LpSolutionOptimal``。HiGHS 下唯一可靠的办法是直接读求解器的 ``mip_gap``。

注意 HiGHS 的**默认** ``mip_rel_gap`` 就是 ``1e-4``，所以"未证明最优"是常态而
非异常——论文必须如实报告达到的间隙，而不是笼统写"最优"。
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# 判定"证明最优"的间隙阈值。HiGHS 默认容差 1e-4 会留下非零 gap，
# 取 1e-9 意味着"必须解到数值精度极限"才算证明。
GAP_TOL = 1e-9


def solver_gap(prob) -> float | None:
    """求解器**实际达到**的相对 MIP 间隙；读不到返回 ``None``。

    纯 LP（无整数变量）在 HiGHS 下 ``mip_gap`` 为 NaN，此时返回 ``None``
    ——调用方应理解为"不存在 MIP 间隙这一概念"。
    """
    sm = getattr(prob, "solverModel", None)
    if sm is None:
        return None
    try:
        g = float(sm.getInfo().mip_gap)
    except Exception:
        return None
    return None if g != g else g          # NaN -> None


def proven_optimal(prob) -> bool:
    """**是否证明了最优性**（而不是"停下且有可行解"）。"""
    import pulp

    if pulp.LpStatus[prob.status] != "Optimal":
        return False

    sm = getattr(prob, "solverModel", None)
    if sm is not None:                                   # HiGHS 路径
        g = solver_gap(prob)
        if g is not None:
            return g <= GAP_TOL
        # 读不到 gap（NaN=纯 LP）就退到 sol_status，至少不比原来差

    return getattr(prob, "sol_status", None) == pulp.constants.LpSolutionOptimal


def describe(prob, label: str = "") -> tuple[str, float | None]:
    """返回 ``(状态标签, 实际间隙)``，并**按事实**打日志。

    状态标签只可能是：

    * ``"Optimal"`` —— 证明了最优（gap <= 1e-9）
    * ``"Feasible"`` —— 有可行解但未证明最优（达到间隙容差或时限）

    早期若干处把"达间隙而停"记成 ``"Optimal"``，论文据此写"精确最优"，
    属于不可追溯的声称。此处统一口径。
    """
    proven = proven_optimal(prob)
    gap = solver_gap(prob)
    tag = "Optimal" if proven else "Feasible"
    if proven:
        logger.info("[%s] 已证明最优（相对间隙 <= %.0e）", label, GAP_TOL)
    else:
        logger.warning(
            "[%s] 未证明最优：相对间隙 %s —— 返回当前最好解。"
            "论文引用该结果时须写成「求解至相对间隙 X%%」而非「最优」。",
            label,
            ("%.4f%%" % (100.0 * gap)) if gap is not None else "不可读（可能为纯 LP）",
        )
    return tag, gap
