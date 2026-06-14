# Lane-Emden 方程数值研究 — 项目文件参考手册

> 更新时间：2026-06-14（目录重组 + 清理bug模块）

---

## 项目目录结构

```
小课题/
├── data_input.py                  # 数据中枢（共享依赖）
├── run_all_experiments.py         # 统一实验入口
├── PROJECT_REFERENCE.md           # 本文档
├── output_initial/                # 图片输出目录
│
├── solvers/                       # 核心求解器（IVP + BVP）
│   ├── rk4.py                     # 经典RK4求解器
│   ├── fd.py                      # 二阶FD + 阻尼Newton
│   ├── adaptive.py                # 自适应DP5(4)
│   └── implicit.py                # 隐式Radau IIA
│
├── physics/                       # 物理拓展
│   ├── isothermal.py              # 等温球体 (n→∞)
│   ├── geometry.py                # Slab/Cylinder/Sphere几何
│   ├── tov.py                     # 相对论TOV方程
│   └── quantities.py              # 物理量系统计算
│
└── analysis/                      # 分析与验证
    ├── initial_guess.py           # ε/h敏感性研究
    ├── richardson.py              # Richardson外推
    ├── spectral.py                # Chebyshev谱方法
    ├── manufactured.py            # MMS构造解验证
    ├── shooting.py                # 打靶法
    └── uncertainty.py             # 不确定性传导
```

---

## 一、基础设施层

### `data_input.py` — 数据加载中枢
| 项目 | 说明 |
|------|------|
| **读取数据** | `polytrope_global_properties.csv`（11个n值的ξ₁、θ'(ξ₁)、质量参数等） |
| | `lane_emden_tables_no_page.csv`（Horedt 1986七位表，含Slab/Cylinder/Sphere） |
| **核心类** | `LaneEmdenReferenceData` — 提供 `get_first_zero(n)`、`interpolate_theta(n,xi)` 等 |
| **已知问题** | Horedt表Sphere数据存在严重OCR列错位，theta列不可靠 |

---

## 二、核心求解器（solvers/）

### `solvers/rk4.py` — 经典四阶Runge-Kutta求解器 (IVP)
| 项目 | 说明 |
|------|------|
| **方法** | 固定步长经典RK4，θ(ε)由Taylor展开（O(ξ⁴)）初始化 |
| **输出** | `LaneEmdenSolution`（xi, theta, theta_prime, first_zero） |
| **局限** | 非整数n在θ变负时崩溃；实测收敛阶~2（受限于启动误差） |

### `solvers/fd.py` — 二阶中心差分 + 阻尼Newton求解器 (BVP)
| 项目 | 说明 |
|------|------|
| **方法** | 均匀网格中心差分离散化，三对角Jacobian + Thomas算法，阻尼Newton迭代 |
| **输出** | `FiniteDifferenceSolution`（含converged, iterations, residual_norm） |
| **优势** | 对全范围n∈[0,5]稳定，实测收敛阶≈2 |

### `solvers/adaptive.py` — 自适应步长Dormand-Prince 5(4)
| 项目 | 说明 |
|------|------|
| **方法** | 7-stage嵌入对，5阶解推进 + 4阶解估计局部误差 |
| **步长控制** | `h_new = 0.9 * h * (tol/err)^{1/5}`，含安全因子和步长上下界 |
| **实验** | 对n=0,1对比固定步长RK4 vs 自适应RK5(4)效率（精度 vs RHS评估次数） |
| **输出** | `AdaptiveRKSolution`（含accepted/rejected步数、h范围等统计） |
| **图片** | `adaptive_rk_efficiency.png` |

### `solvers/implicit.py` — 隐式Runge-Kutta (Radau IIA)
| 项目 | 说明 |
|------|------|
| **方法** | 3-stage Radau IIA（5阶，L-stable），简化Newton求解级值方程 |
| **适用场景** | n接近5时方程变stiff，隐式方法允许更大步长 |
| **实验** | 对n=0,3,4.5对比显式RK4和隐式Radau的步长需求 |

---

## 三、物理拓展（physics/）

### `physics/isothermal.py` — 等温球体 (n→∞)
| 项目 | 说明 |
|------|------|
| **方程** | (1/ξ²)d/dξ(ξ²dψ/dξ)=exp(-ψ)，ψ(0)=ψ'(0)=0 |
| **方法** | RK4 + Taylor展开O(ξ⁶)初始化 |
| **实验** | ξ_max∈{10,30,50}，对比数值解与渐近解 |
| **图片** | `isothermal_overview.png` |

### `physics/geometry.py` — 柱对称与平面对称
| 项目 | 说明 |
|------|------|
| **方程** | (1/ξ^k)d/dξ(ξ^k dθ/dξ)+θⁿ=0，k=0(Slab),1(Cylinder),2(Sphere) |
| **方法** | 参数化RK4，k相关的Taylor展开 |
| **实验** | 对比三种几何的ξ₁和θ(ξ)剖面 |
| **图片** | `geometry_overview.png` |

### `physics/tov.py` — 相对论性TOV方程
| 项目 | 说明 |
|------|------|
| **方程** | dθ/dξ=-(θⁿ+1)(m+σξ³θⁿ)/(ξ²(1-2σm/ξ))，dm/dξ=ξ²θⁿ |
| **参数** | σ=P_c/(ρ_c c²)——σ=0退化为标准LE；σ~0.3-0.5为中子星区间 |
| **实验** | σ=0极限验证（与LE对比）；n=1.5的质量-半径曲线（含σ增大方向标注） |
| **图片** | `tov_overview.png`（含Newtonian极限对比表和物理含义说明） |

### `physics/quantities.py` — 物理量系统计算
| 项目 | 说明 |
|------|------|
| **计算量** | ξ₁、θ'(ξ₁)、无量纲质量、中心凝聚度ρ_c/⟨ρ⟩、质量-半径幂律指数 |
| **函数** | `generate_physical_table()`、`compute_mass_radius_relation()` |
| **图片** | `physical_quantities.png` |

---

## 四、分析与验证（analysis/）

### `analysis/initial_guess.py` — ε初值与步长敏感性系统研究
| 项目 | 说明 |
|------|------|
| **实验1** | 精确解情形（n=0,1,5）：ε∈{1e-3,1e-4,1e-5} × h∈{2e-2,...,2.5e-3} |
| **实验2** | 无解析解情形（n=0.5~4.5）：在ξ₁/4, ξ₁/2, 3ξ₁/4及θ'(ξ₁)处比较 |
| **参照策略** | ξ₁和θ'(ξ₁)←外部CSV；中间节点θ←高分辨率FD（h=5e-4）；Horedt表因OCR损坏未采用 |
| **输出** | 4个CSV + 3张PNG + 分析摘要 |

### `analysis/richardson.py` — Richardson外推
| 项目 | 说明 |
|------|------|
| **功能** | 利用h和h/2两套网格做Richardson外推：RK4用p=4权重，FD用p=2权重 |
| **实验** | 对n=0,1,5分别测试RK4和FD外推前后误差与收敛阶 |
| **关键发现** | 外推提升有限（启动误差主导，非纯O(h^p)），已在图中标注 |
| **图片** | `richardson_convergence.png` |

### `analysis/spectral.py` — Chebyshev谱方法
| 项目 | 说明 |
|------|------|
| **方法** | CGL配点 + 偶延拓（在[-ξ_max, ξ_max]上求解规避ξ=0奇性） |
| **优势** | 指数收敛——N=30可达10⁻¹⁰精度；偶数N收敛（偶延拓保持），奇数N发散 |
| **图片** | `spectral_convergence.png`（含节点分布示意和方法说明） |

### `analysis/manufactured.py` — 构造解验证 (MMS)
| 项目 | 说明 |
|------|------|
| **原理** | 选取光滑函数θ_man(ξ)满足BC → 计算源项S(ξ) → 求解带源项方程 → 与θ_man比较 |
| **预设解** | CosineBump、ExponentialDecay、PolynomialBump |
| **关键发现** | 实测阶≈2.0——确认启动误差主导（而非方法实现错误） |
| **图片** | `manufactured_convergence.png` |

### `analysis/shooting.py` — 打靶法
| 项目 | 说明 |
|------|------|
| **方法** | RK4推进 + 割线法调整ξ₁使θ(ξ₁)=0 |
| **精度** | 1-3次迭代收敛，ξ₁误差~10⁻⁹ |

### `analysis/uncertainty.py` — 不确定性传导分析
| 项目 | 说明 |
|------|------|
| **方法** | 多分辨率求解 → Richardson外推 → 估计标准不确定度 + GCI |
| **输出** | 每个物理量的best value、±uncertainty、relative uncertainty、observed order |

---

## 五、统一入口

### `run_all_experiments.py` — 统一实验入口
| 项目 | 说明 |
|------|------|
| **使用** | `python run_all_experiments.py [--quick]` |
| **7个实验** | 基本收敛→物理量→几何对比→MMS验证→方法对比→等温球→TOV |
| **输出** | `output/experiment_summary.txt` |

---

## 六、已删除模块（KILL）

| 文件 | 原因 |
|------|------|
| `fd_high_order.py` | Newton不收敛，banded Jacobian构造有bug |
| `continuation.py` | n=2出现NaN，\|θ\|ⁿsign(θ)对非整数n不稳定 |
| `rotating.py` | λ>0全部崩溃，仅λ=0有效（=标准LE），无实质功能 |
| `parameter_continuation.py` | 功能与quantities.py重复，实现有monkey-patching hack |

---

## 七、文件依赖关系

```
data_input.py          ← 被所有模块引用（数据中枢）
solvers/rk4.py         ← 被 adaptive, implicit, shooting, richardson, initial_guess 调用
solvers/fd.py          ← 被 quantities, uncertainty, initial_guess, richardson 调用

analysis/richardson.py       → 依赖 solvers/rk4.py, solvers/fd.py
analysis/initial_guess.py    → 依赖 solvers/rk4.py, solvers/fd.py, data_input.py
analysis/uncertainty.py      → 依赖 solvers/fd.py
physics/quantities.py        → 依赖 solvers/fd.py, data_input.py

其余文件（adaptive, implicit, spectral, manufactured, shooting,
          isothermal, geometry, tov）→ 仅依赖 data_input.py
```
