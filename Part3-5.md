## 第三部分、具体工作流程

```mermaid
flowchart LR
    A[data_input.py] --> A1[读取解析解与参考表]
    A --> A2[提供 xi_1 / 表面导数 / 参考节点]

    B[Ronge-Kutta.py] --> B1[泰勒初始化]
    B --> B2[RK4 推进求解 theta(xi)]

    C[finite-difference.py] --> C1[网格离散]
    C --> C2[Newton 迭代]
    C --> C3[Thomas 算法解三对角线性方程组]

    D[initial_guess.py] --> D1[调用 data_input.py]
    D --> D2[调用 Ronge-Kutta.py]
    D --> D3[调用 finite-difference.py]
    D --> D4[固定 h: 分析 epsilon 对误差的影响]
    D --> D5[固定最优 epsilon: 分析 h 对误差与收敛阶的影响]
    D --> D6[生成 overview 图与分析文本]

    A1 --> D
    A2 --> D
    B2 --> D
    C3 --> D

    D --> E[output_initial/]
    E --> E1[exact_epsilon_study.csv]
    E --> E2[exact_h_study.csv]
    E --> E3[special_nodes_epsilon_study.csv]
    E --> E4[special_nodes_h_study.csv]
    E --> E5[exact_overview.png]
    E --> E6[special_nodes_overview.png]
    E --> E7[theta_n_overview.png]
    E --> E8[initialization_convergence.txt]
    E --> E9[analysis_summary.txt]
```
