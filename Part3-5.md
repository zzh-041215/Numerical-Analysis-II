## 第三部分、具体工作流程
### 3.1、工作流程
我在图表当中展现了这个项目实现流程，大致实现框架并不复杂：主要就是从选题、文献与数据集查找、数据清洗、有限差分和Ronge-Kutta四阶算法的数值求解、初始解寻找、方程求解这些步骤。而最后的结果将进一步展现在`output_initial`这个子文件夹中，工作流程如下:

```mermaid
小课题
    Horedt, G. P. “Seven-digit tables of Lane-Emden functions.” Astrophysics and Space Science, 1986.pdf #关于Lame-Enden方程各种集合情况下的数值表格
    lane_emden_tables_no_pages.csv #从上述PDF中清洗出来的具体数据细节
    polytrope_global_properties.csv #从天文网站中下载下来的关于几个常见n值的数据基本信息(如零点、表面处导数值等)
    data_input.py #数据清洗脚本，将上述两个csv当中的数据转化为python中可以直接调用的表格
    finite_difference.py #有限差分算法的数值求解脚本
    runge_kutta.py #Ronge-Kutta四阶算法的数值求解
    initial_guess.py #主脚本，用来分析不同初始值和步长条件下函数的数值解的行为，并和解析解做出细致比较
    output_initial #结果输出文件夹，包括图片表格和日志
        analysis_summary.txt #打印程序输出的日志文件
        exact_epsilon_study.csv #对于解析解情况，分析误差无穷范数和收敛情况
        exact_h_study.csv #误差和步长关系
        exact_overview.png #对于解析解情况的数值解和解析解的对比图
        special_nodes_epsilon_study.csv #对于没有解析解情况，分析误差和初始$\xi$的关系
        special_nodes_h_study.csv #对于没有解析解情况，分析误差和步长的关系
        special_nodes_overview.png #误差随h变化的总览图
        theta_n_overview.png #全局数值解随n变化的总览图
```
### 3.2、输出结果分析

对于n=0,1,5这三个有解析解的方程而言，每一个情况中首先固定一个`h`值改变$\xi$的初值，比较定义域当中数值解与解析解之间误差的无穷范数:
$$
E_{\infty}=\max_{j}|\theta_{num}(\xi_{j})-\theta_{ref}(\xi_{j})|
$$
通过判断此时最小的$E_{\infty}$对应的$\xi$值作为合理的初值选取，尽可能地降低微分方程零点处奇性对求解的影响。在这一步的基础上，固定该$\xi$改变`h`值，比较$E_{\infty}$随着步长的变化情况，将两种算法在相同步长的情况下做出误差比较，分析数值稳定性，具体图片如下:
![exact_overview](./output_initial/exact_overview.png)
可以看出来在这些算法的数值解情况下，初始值的选取对于误差的影响并不算很大，不过几种情况下都是$\xi=10^{-3}$的初始值相对最为稳定。而对于步长而言，误差总是随着步长的增大而增大。这也很好理解，因为步长越大函数就相对而言越为离散。

而对于其他没有解析解的n值(如n=1.5,3等)情况而言，我们主要比较一些关键点，如零点、表面导数等。这些数据在`polytrope_global_properties.csv`当中也有所体现。在此处我们同样沿用上述给出的带有解析解的n值的初始$\xi$选取方法，继续改变h，判断数值解出来的零点、表面导数和官网上的真实数据之间的关联。
![special_nodes_overview](./output_initial/special_nodes_overview.png)
可以看到，对比而言一般来说RK4算法都更为稳定，误差更小一点，但是面对一些特殊情况时也容易崩溃，比如n=1.5,2.5的情况较难计算出来。而有限差分算法误差可能会更大，但算法的稳定性也会相应更强。后面的章节会进一步分析它们的数值稳定性和计算复杂度。

同时我们也生成了一张完整的函数图像，展示了不同n值的$\theta(\xi)$的宏观数值解情况:
![theta_n_overview](./output_initial/theta_n_overview.png)
可以看到随着n值的增大，函数的下降速率也在变慢，零点在向右移动。在临界情况下n=5，$\theta=\frac{1}{\sqrt{1+\xi^{2}/3}}$，零点在无穷远处。

## 第四部分、误差分析

### 4.1、RK4收敛阶数简析

RK4主要是对Taylor展开的高阶项进行加权平均，天然适合初值问题IVP。对于初值问题 $y'=f(t,y)$，单步更新公式为：
$$
y_{n+1}=y_n+\frac{h}{6}(k_1+2k_2+2k_3+k_4)
$$

**局部截断误差**：考虑单步从 $y_n$ 到 $y_{n+1}$ 的误差。假设 $y_n$ 是精确的，则局部截断误差定义为：
$$
\tau_{n+1}=y(t_{n+1})-y_{n+1}
$$
将真实解 $y(t_{n+1})$ 在 $t_n$ 处进行Taylor展开：
$$
y(t_{n+1})=y(t_n)+hy'(t_n)+\frac{h^2}{2}y''(t_n)+\frac{h^3}{6}y'''(t_n)+\frac{h^4}{24}y^{(4)}(t_n)+O(h^5)
$$
将后一项针对前一项展开并作系数加减，发现恰好能够匹配Taylor展开式中直到 $h^4$ 项的系数，因此局部截断误差为：
$$
\tau_{n+1}=O(h^5)
$$
这意味着单步误差与步长的五次方成正比。

**全局误差**：考虑从初值 $t_0$ 到终点 $t_N$ 的累积误差。设总区间长度为 $L=t_N-t_0$，步数为 $N=L/h$。每一步的局部误差 $O(h^5)$ 会累积 $N$ 次，因此全局误差为：
$$
E_{global}=N\cdot O(h^5)=\frac{L}{h}\cdot O(h^5)=O(h^4)
$$
可以看到全局的RK4求解微分方程的数值阶数能够稳定到四阶，只要当$\xi>\epsilon>0$，且函数足够光滑的情况下，四阶的收敛准则天然成立。该算法总体来说计算较为简便、收敛速率相对较快，但是对于初值较为敏感### 4.2、有限差分算法收敛阶数
首先考虑单步误差
$$
\theta_{i+1}=\theta_{i}+h\theta_{i}'+\frac{h^{2}}{2}\theta_{i}''+\frac{h^{3}}{6}\theta_i'''+\frac{h^{4}}{24}\theta_i^{(4)}+O(h^{5})\\
\theta_{i-1}=\theta_{i}-h\theta_i'+\frac{h^{2}}{2}\theta_{i}''-\frac{h^{3}}{6}\theta_i'''+\frac{h^{4}}{24}\theta_i^{(4)}+O(h^{5})\\
\frac{\theta_{i+1}-2\theta_{i}+\theta_{i-1}}{h^{2}}\sim \theta_{i}''+\frac{h^{2}}{12}\theta_{i}^{(4)}\\
\text{同理},\,\frac{\theta_{i+1}-\theta_{i-1}}{2h}\sim \theta_{i}'+\frac{h^{2}}{6}\theta_{i}'''
$$
其次考虑真实解情形，令数值解为U，真实解的离散情况为u，$J_{h}$为Jacobi矩阵：
$$
e=U-u,\quad J_{h}e=\tau,\quad e=J_{h}^{-1}\tau\\
\|e\|\leq\|J_{h}^{-1}\|\|\tau\|\leq Ch^{2}=O(h^{2})
$$
## 第五部分、算法复杂度

### 5.1、RK4
在RK4算法当中将二阶线性微分方程问题转换为了一个基本的初值问题。上文已经有所表现:
$$
k_1=f(\xi_n,y_n),\quad k_2=f(\xi_n+\frac{h}{2}k_1)\\
k_3=f(\xi_n+\frac{h}{2},y_n+\frac{h}{2}k_2),\quad k_4=f(\xi_n+h,y_n+hk_3)\\
y_{n+1}=y_{n}+\frac{h}{6}(k_1+2k_2+2k_3+k_4)
$$
在不断外推之后就可以完成$(\theta_n,\theta_n')\rightarrow(\theta_{n+1},\theta_{n+1}')$的更新，而每一步的运算都是常数量级的，因此总步数是$O(N)$量级的。

同时在Lane-Emden问题上，我们还关心第一个零点的位置，即$\theta(\xi)=0$, 物理上对应恒星边界的无量纲边界。考虑到取点的`h`足够之小，因此在符号出现变异性的时候采取线性插值即可:
$$
\xi_1\approx \xi_{n}+\frac{|\theta_{n}|}{\theta_{n}+\theta_{n+1}}(\xi_{n+1}-\xi_{n})
$$
整个过程可以理解为:
```text
Lane-Emden ODE
    ↓
改写成一阶系统
    ↓
Taylor展开处理中心奇点
    ↓
从 ε 开始、RK4逐步推进
    ↓
检测 θ 是否穿过0
    ↓
插值求 ξ1
```
该过程空间和时间复杂度都为O(N),运算上较为轻量级。

### 5.2、有限差分算法
和RK4算法在思维上有显著不同的是，有限差分考虑的是较为宏观的情形。建立一个较大的网格$\xi_0,\ldots,\xi_n,\quad h=(\xi_n-\epsilon)/N$,随后将连续问题离散化，求解这些点上的$\theta$值:
$$
\frac{\theta_{i+1}-2\theta_{i}+\theta(i-1)}{h^{2}}+\frac{\theta_{i+1}-\theta{i-1}}{\xi_i h}+\theta_{i}^{n}=0
$$
这个离散问题可以转化为一个求解非线性方程组的问题:
$$
F(\theta)=0,\quad J=\frac{\partial F}{\partial \theta}\\
F(\theta^{*}) \approx F(\theta^{(k)})+\nabla F(\theta^{(k)})(\theta^{*}-\theta^{(k)})\\
\theta^{(k+1)}=\theta^{(k)}-(\nabla F(\theta^{(k)}))^{-1}F(\theta^{(k)})
$$
每一步仅需要求解一个线性方程组就可以了，一般情况下这是一个$O(N^{3})$的时间复杂度算法，所幸这是一个三对角矩阵，仅仅需要通过Thomas算法便可以实现，从上至下消元消除下三角元，而成为上三角矩阵之后消除上对角元即可，单步复杂度为$O(N)$。

整个流程如下即可:
```text
Lane-Emden ODE并区间整体离散
    ↓
中心差分近似导数，得到非线性代数方程组
    ↓
构建Jacobian矩阵并给出的矩阵形式更新
    ↓
Thomas算法解三对角系统
    ↓
阻尼更新
    ↓
直到残差收敛停机
```
假设迭代过程为K步，那么总算法复杂度为$O(KN)$,$K<<N$,因此该过程的算法复杂度也为$O(N)$，就是相较而言系数更大。