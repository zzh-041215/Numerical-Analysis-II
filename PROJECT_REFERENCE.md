# Lane-Emden 方程数值研究 — 项目文件参考手册

> 更新时间：2026-06-14

---

## 项目总览

本项目研究 Lane-Emden 方程的数值求解，涵盖标准方程及其多种拓展形式。方法体系包括显式/隐式 Runge-Kutta、有限差分、谱方法、打靶法等。对比参照包括解析解（n=0,1,5）、外部参考数据（polytrope_global_properties.csv）和数值交叉验证。

```
求解器层 (solvers)        验证层 (verification)      拓展层 (extensions)
─────────────────────    ─────────────────────     ─────────────────────
Ronge-Kutta.py           richardson.py             isothermal.py
finite-difference.py     manufactured.py           generalized_geometry.py
adaptive_rk.py           physical_quantities.py    tov.py
fd_high_order.py         uncertainty.py            rotating.py
spectral.py              shooting.py               continuation.py
implicit_rk.py           parameter_continuation.py
```

---

## 一、基础设施层（原始项目文件）

### 1. `data_input.py` — 数据加载中枢
| 项目 | 说明 |
|------|------|
| **读取数据** | `polytrope_global_properties.csv`（11个n值的ξ₁、θ'(ξ₁)、质量参数等） |
| | `lane_emden_tables_no_page.csv`（Horedt 1986七位表，含Slab/Cylinder/Sphere三种几何） |
| **核心类** | `LaneEmdenReferenceData` — 提供 `get_first_zero(n)`、`interpolate_theta(n,xi)` 等查询接口 |
| **已知问题** | Horedt表Sphere数据存在严重OCR列错位，theta列不可靠（详见 `initial_guess.py` 数据溯源章节） |

### 2. `Ronge-Kutta.py` — 经典四阶Runge-Kutta求解器 (IVP)
| 项目 | 说明 |
|------|------|
| **方法** | 固定步长经典RK4，θ(ε)由Taylor展开（O(ξ⁴)）初始化 |
| **输出** | `LaneEmdenSolution`（xi, theta, theta', first_zero） |
| **局限** | 非整数n在θ变负时崩溃；实测收敛阶为~2（受限于启动误差） |

### 3. `finite-difference.py` — 二阶中心差分 + 阻尼Newton求解器 (BVP)
| 项目 | 说明 |
|------|------|
| **方法** | 均匀网格中心差分离散化，三对角Jacobian + Thomas算法，阻尼Newton迭代 |
| **输出** | `FiniteDifferenceSolution`（含converged, iterations, residual_norm等收敛信息） |
| **优势** | 对全范围n∈[0,5]稳定，实测收敛阶≈2 |

### 4. `initial_guess.py` — ε初值与步长敏感性系统研究
| 项目 | 说明 |
|------|------|
| **实验1** | 精确解情形（n=0,1,5）：ε∈{1e-3,1e-4,1e-5} × h∈{2e-2,...,2.5e-3} |
| **实验2** | 无解析解情形（n=0.5~4.5）：在ξ₁/4, ξ₁/2, 3ξ₁/4及θ'(ξ₁)处比较 |
| **参照策略** | ξ₁和θ'(ξ₁)←外部CSV；中间节点θ←高分辨率FD（h=5e-4）；Horedt表因OCR损坏未采用（见§3数据溯源） |
| **输出** | `output_initial/`下4个CSV + 3张PNG + 分析摘要 |

---

## 二、模块A：高阶数值精度验证

### 5. `richardson.py` — Richardson外推
| 项目 | 说明 |
|------|------|
| **功能** | 利用h和h/2两套网格做Richardson外推：RK4用p=4权重，FD用p=2权重 |
| **核心函数** | `richardson_extrapolate_rk4()`, `richardson_extrapolate_fd()` |
| | `verify_richardson_order()` — 系统验证外推前后收敛阶变化 |
| | `compute_convergence_order()` — log-log线性回归 + R²置信度 |
| **实验** | 对n=0,1,5分别测试RK4和FD外推前后的误差与收敛阶 |

### 6. `fd_high_order.py` — 四阶紧致有限差分
| 项目 | 说明 |
|------|------|
| **方法** | 五节点模板四阶中心差分，内点O(h⁴)，近边界点回退到O(h²) |
| **Jacobian** | 五对角带状矩阵，用`scipy.linalg.solve_banded`求解 |
| **实验** | 对n=0,1,3用N=100/200/400/800对比二阶和四阶FD精度 |

### 7. `adaptive_rk.py` — 自适应步长Dormand-Prince 5(4)
| 项目 | 说明 |
|------|------|
| **方法** | 7-stage嵌入对，5阶解推进 + 4阶解估计局部误差 |
| **步长控制** | `h_new = 0.9 * h * (tol/err)^{1/5}`，含安全因子和步长上下界 |
| **实验** | 对n=0,1,3对比固定步长RK4（多h值）vs自适应RK5(4)（多tol值） |
| **输出** | `AdaptiveRKSolution`（含accepted/rejected步数、h范围等统计） |

### 8. `spectral.py` — Chebyshev谱方法
| 项目 | 说明 |
|------|------|
| **方法** | Chebyshev-Gauss-Lobatto配点 + 偶延拓（在[-ξ_max, ξ_max]上求解以避免ξ=0奇性） |
| **Newton迭代** | 全矩阵Newton（N≤50时可行），阻尼更新保证正性 |
| **优势** | 指数收敛——N=30可达10⁻¹⁰精度 |
| **实验** | 对n=0,1,3用N=20/30/40/50测试收敛行为 |

### 9. `manufactured.py` — 构造解验证 (MMS)
| 项目 | 说明 |
|------|------|
| **原理** | 选取光滑函数θ_man(ξ)满足BC，计算源项S(ξ)=θ_man''+2θ_man'/ξ+θ_manⁿ，求解带源项方程后与θ_man比较 |
| **预设解** | `CosineBumpSolution`（cos波包）、`ExponentialDecaySolution`（指数衰减）、`PolynomialBumpSolution`（多项式） |
| **实验** | 对任意n（含非整数n）精确验证收敛阶 |
| **关键发现** | 实测阶≈2.0而非4.0——确认启动误差主导 |

---

## 三、模块B：方程拓展情形

### 10. `isothermal.py` — 等温球体 (n→∞)
| 项目 | 说明 |
|------|------|
| **方程** | (1/ξ²)d/dξ(ξ²dψ/dξ)=exp(-ψ)，ψ(0)=ψ'(0)=0 |
| **方法** | RK4（主求解器）+ FD（辅助），Taylor展开O(ξ⁶)初始化 |
| **渐近解** | ψ~ln(ξ²/2)-2ln(ln(ξ²/2))（大ξ验证） |
| **实验** | ξ_max∈{10,30,50}，对比数值解与渐近解，生成密度剖面图 |

### 11. `generalized_geometry.py` — 柱对称与平面对称
| 项目 | 说明 |
|------|------|
| **方程** | (1/ξ^k)d/dξ(ξ^k dθ/dξ)+θⁿ=0，k=0(Slab),1(Cylinder),2(Sphere) |
| **方法** | 参数化的RK4求解器，k相关的Taylor展开 |
| **实验** | 对n=1.5对比三种几何的ξ₁和θ(ξ)剖面；对比Horedt表中对应几何数据 |
| **输出** | 几何对比图 + 球对称多n值总览图 |

### 12. `tov.py` — 相对论性TOV方程
| 项目 | 说明 |
|------|------|
| **方程** | dθ/dξ=-(m+σξ³θⁿ)/(ξ²(1-2σm/ξ))，dm/dξ=ξ²θⁿ |
| **参数** | σ=P_c/(ρ_c c²)——σ→0退化为标准LE |
| **实验** | σ=0极限验证；n=1.5质量-半径曲线；不同σ下的密度剖面 |

### 13. `rotating.py` — 旋转多方球
| 项目 | 说明 |
|------|------|
| **方程** | (1/ξ²)d/dξ(ξ²dθ/dξ)+θⁿ=λ，λ=ω²/(2πGρ_c) |
| **实验** | λ∈[0,0.4]对n=1.5的半径膨胀效应；多n值的R(λ)曲线 |

---

## 四、模块C：长程求解

### 14. `continuation.py` — 过零点延拓
| 项目 | 说明 |
|------|------|
| **方法** | |θ|ⁿ·sign(θ)替代θⁿ（物理延拓），允许穿越第一零点继续积分 |
| **实验** | 对n=0,1,2追踪前5个零点位置；生成零点位置vs n的关系图 |
| **发现** | n=1（sin(ξ)/ξ）可找到无限多个零点（ξ=nπ），n=2在过零后θ→NaN |

### 15. `implicit_rk.py` — 隐式Runge-Kutta (Radau IIA)
| 项目 | 说明 |
|------|------|
| **方法** | 3-stage Radau IIA（5阶，L-stable），简化Newton求解级值方程 |
| **适用场景** | n接近5时方程变stiff，隐式方法允许更大步长 |
| **实验** | 对n=0,3,4.5对比显式RK4和隐式Radau的步长需求 |

### 16. `shooting.py` — 打靶法
| 项目 | 说明 |
|------|------|
| **方法** | RK4推进 + 割线法调整ξ₁使θ(ξ₁)=0 |
| **实验** | 对n=0,1,2,3打靶求解，与参考ξ₁对比误差；展示经典打靶（调整θ'(ε)） |
| **精度** | 1-3次迭代收敛，ξ₁误差~10⁻⁹ |

### 17. `parameter_continuation.py` — 参数延拓
| 项目 | 说明 |
|------|------|
| **方法** | 从n=0出发，Δn递增，前一个解作为下一个的Newton初值 |
| **实验** | 追踪ξ₁(n)从n=0到n=4.5的演化；分析n→5临界发散行为 |
| **标度律** | ξ₁~(5-n)^α，拟合α与理论值-0.5比较 |

---

## 五、模块D：物理分析

### 18. `physical_quantities.py` — 物理量系统计算
| 项目 | 说明 |
|------|------|
| **计算量** | ξ₁、θ'(ξ₁)、无量纲质量-ξ₁²θ'(ξ₁)、中心凝聚度ρ_c/⟨ρ⟩、质量-半径幂律指数 |
| **函数** | `generate_physical_table()` — 生成全n值物理量表格 |
| | `compute_mass_radius_relation()` — 给定K和ρ_c计算绝对M和R |

### 19. `uncertainty.py` — 不确定性传导分析
| 项目 | 说明 |
|------|------|
| **方法** | 多分辨率求解 → Richardson外推 → 估计标准不确定度 + GCI |
| **输出** | 每个物理量的best value、±uncertainty、relative uncertainty、observed order |

---

## 六、统一入口

### 20. `run_all_experiments.py` — 统一实验入口
| 项目 | 说明 |
|------|------|
| **使用** | `python run_all_experiments.py [--quick]` |
| **8个实验** | 基本收敛→物理量→几何对比→MMS验证→方法对比→等温球→TOV→参数延拓 |
| **输出** | `output/experiment_summary.txt` |

---

## 文件依赖关系

```
data_input.py          ← 被所有模块引用（数据中枢）
Ronge-Kutta.py         ← 被 richardson, adaptive_rk, shooting, run_all 调用
finite-difference.py   ← 被 physical_quantities, uncertainty, parameter_continuation, initial_guess 调用

richardson.py          → 依赖 Ronge-Kutta.py, finite-difference.py
adaptive_rk.py         → 依赖 data_input.py
fd_high_order.py       → 独立（自包含FD求解器）
spectral.py            → 独立（自包含谱求解器）
manufactured.py        → 独立（自包含MMS框架）
isothermal.py          → 独立（自包含等温求解器）
generalized_geometry.py → 依赖 data_input.py
tov.py                 → 独立（自包含TOV求解器）
rotating.py            → 独立（自包含旋转求解器）
continuation.py        → 独立（自包含延拓求解器）
implicit_rk.py         → 依赖 data_input.py
shooting.py            → 依赖 data_input.py
parameter_continuation.py → 依赖 finite-difference.py
physical_quantities.py → 依赖 data_input.py, finite-difference.py
uncertainty.py         → 依赖 physical_quantities.py, finite-difference.py
initial_guess.py       → 依赖 Ronge-Kutta.py, finite-difference.py, data_input.py
run_all_experiments.py → 依赖上述大部分模块
```
