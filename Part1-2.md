## 第一部分:背景知识介绍与研究挑战
### 1.1 Lane-Emden方程的推导
考虑到自引力势能、流体静力平衡下的球对称球体和质量守恒，我们可以使用以下方程:
$$
\frac{dm}{dr}=4\pi r^{2}\rho
$$
而流体内部存在一个平衡，在给定半径的时候取一个面积足够小的面元，那么它受到的内部恒星的引力应当和内外部恒星的压力差达到平衡，即:
$$
dF_{p}=P(r)4\pi r^{2}-P(r+dr)4\pi r^{2}\\
P(r+dr)=P(r)+\frac{dP}{dr}dr,\rightarrow dF_{p}=-\frac{dP}{dr}4\pi r^{2}dr\\
dF_{G}=-\frac{Gm(r)}{r^{2}}dm=-4\pi Gm(r)\rho dr\\
dF_{P}+dF_{G}=0\rightarrow 
-\frac{dP}{dr}4\pi r^{2}dr-4\pi Gm(r)\rho dr=0\\
\frac{1}{\rho}\frac{dP}{dr}=-\frac{Gm(r)}{r^{2}}
$$
m也是r的函数，再做一次求导数可以得到:
$$
\frac{d}{dr}(\frac{1}{\rho}\frac{dP}{dr})=\frac{2Gm}{r^{3}}-\frac{G}{r^{2}}\frac{dm}{dr}=-\frac{2}{\rho r}\frac{dP}{dr}-4\pi G\rho\\
r^{2}\frac{d}{dr}(\frac{1}{\rho}\frac{dP}{dr})+\frac{2r}{\rho}\frac{dP}{dr}=\frac{d}{dr}(\frac{r^{2}}{\rho}\frac{dP}{dr})=-4\pi G\rho r^{2}
$$
根据多变方程，我们可以进一步化简恒星内部压强，而根据密度关系，我们可以分离出无量纲的密度参数$\theta$:
$$
P=K\rho_{c}^{1+\frac{1}{n}}\theta^{n+1},\quad \rho=\rho_{c}\theta^{n}\\
\frac{1}{r^{2}}\frac{d}{dr}(r^{2}K\rho_{c}^{1/n}(n+1)
\frac{d\theta}{dr})=-4\pi G\rho_{c}\theta^{n}
$$
重新整理常数项，令$\xi=\alpha r$并作整理可以得到:
$$
\frac{1}{\xi^{2}}\frac{d}{d\xi}(\xi^{2}\frac{d\theta}{d\xi})+\theta^{n}=0\\
\alpha^{2}=\frac{4\pi G}{K(n+1)}\rho_{c}^{1-\frac{1}{n}}
$$
于是便得到了Lane-Emden方程的表达形式，它表达了从恒星中心向外，密度如何随着半径下降，并且存在基本的中心条件:
$$
\theta(0)=1,\quad \theta'(0)=0
$$
### 1.2 Lane-Emden方程的数值问题挑战
我们首先对这个方程展开可以得:
$$
\theta''+\frac{2}{\xi}\theta'+\theta^{n}=0
$$
不过该方程有以下挑战:
1. 在$\xi_i=0$这一处存在奇点，在直接带入的过程中不可以直接把零点代入，需要选取一个$\epsilon$做泰勒展开开始启动计算。
2. 这个方程当且仅当$n=0,1,5$时存在解析解，因此对于一般情况，我们需要数值逼近算法来求解这个问题，对于算法的稳定性、收敛性和精度要求相对较高。
3. 很多物理问题关心第一个零点对应的边界条件，也就是$\theta(\xi_1)=0$对应的第一个零点位置$\xi_1$，而对于没有解析解的函数来说，寻找这个点的唯一性问题有待考证，同时数值方法的误差也会影响零点位置的准确性。

为了避免中心奇点带来的数值不稳定，同时也为了比较不同离散方法的精度与效率，本文主要考虑两类数值解法：一类是基于初值问题推进的Runge-Kutta方法，另一类是将区间整体离散后的有限差分方法。

## 第二部分：数值实验设计与实现
### 2.1 基于Runge-Kutta的初值问题解法

令$y_{1}=\theta,\quad y_{2}=\theta',$
则Lane-Emden方程可以写成一阶系统:
$$
\begin{cases}
y_{1}'=y_{2},\\
y_{2}'=-\frac{2}{\xi}y_{2}-y_{1}^{n}.
\end{cases}
$$
由于右端在$\xi=0$处含有$\frac{2}{\xi}$，不能直接从原点代入，考虑到天体物理当中的球对称假设，我们很容易得到$\theta(\xi)$为偶函数，且根据边界条件，在$\xi=0$附近作Taylor展开，可得
$$
\theta(0)=1,\quad \theta'(0)=0\\
\theta(\xi)=1+a\xi^{2}+b\xi^{4}+\ldots\\
\theta'(\xi)=2a\xi+4b\xi^{3}+6c\xi^{5}+\ldots,\quad \theta''(\xi)=2a+12b\xi^{3}+30c\xi^{5}\\
(1+6a)+(na+20b)\xi^{2}+O(\xi^{4})=0\rightarrow a=-\frac{1}{6},\quad b=\frac{n}{120}\\
\theta(\xi)=1-\frac{\xi^{2}}{6}+\frac{n\xi^{4}}{120}+O(\xi^{6}),\quad
\theta'(\xi)=-\frac{\xi}{3}+\frac{n\xi^{3}}{30}+O(\xi^{5}).
$$
因此实际计算时选取一个很小的$\varepsilon>0$作为起点，用上述展开给出
$$
\theta(\varepsilon)\approx 1-\frac{\varepsilon^{2}}{6}+\frac{n\varepsilon^{4}}{120},\quad
\theta'(\varepsilon)\approx -\frac{\varepsilon}{3}+\frac{n\varepsilon^{3}}{30},
$$
再从$\xi=\varepsilon$开始向外推进。

在固定步长情形下，可以使用经典四阶Runge-Kutta方法。若记
$$
Y=(y_{1},y_{2})^{T},\quad F(\xi,Y)=
\begin{pmatrix}
y_{2}\\
-\frac{2}{\xi}y_{2}-y_{1}^{n}
\end{pmatrix},
$$
则每一步的更新为
$$
\begin{aligned}
k_{1}&=F(\xi_{j},Y_{j}),\\
k_{2}&=F(\xi_{j}+\frac{h}{2},Y_{j}+\frac{h}{2}k_{1}),\\
k_{3}&=F(\xi_{j}+\frac{h}{2},Y_{j}+\frac{h}{2}k_{2}),\\
k_{4}&=F(\xi_{j}+h,Y_{j}+hk_{3}),\\
Y_{j+1}&=Y_{j}+\frac{h}{6}(k_{1}+2k_{2}+2k_{3}+k_{4}).
\end{aligned}
$$
该方法实现简单、局部截断误差为$O(h^{5})$，整体误差为$O(h^{4})$，适合用于检验收敛阶。

进一步地，为了兼顾中心附近的精度和远离中心后的计算效率，可以引入自适应步长。基本思想是用一次步长为$h$的计算结果$Y_{h}$与两次步长为$\frac{h}{2}$的计算结果$Y_{h/2}$进行比较，以$E=\|Y_{h/2}-Y_{h}\|$估计局部误差。当$E$小于给定容许误差$\mathrm{tol}$时接受该步，并可适当增大步长；当$E$过大时拒绝该步并减小步长。常用的步长更新形式为:
$$
h_{\mathrm{new}}=s h\left(\frac{\mathrm{tol}}{E}\right)^{1/5},
$$
其中$s\in(0,1)$为安全因子。这样可以在解变化较快的区域自动加密网格，在解较平缓的区域减少不必要的计算。

### 2.2 有限差分方法

除了将方程看作初值问题逐步推进，也可以在有限区间$[0,\xi_{\max}]$上整体离散。取网格点
$$
\xi_{j}=jh,\quad j=0,1,\cdots,N,\quad h=\frac{\xi_{\max}}{N},
$$
并记$\theta_{j}\approx\theta(\xi_{j})$。对内部点$j=1,\cdots,N-1$，使用中心差分近似:
$$
\theta'(\xi_{j})\approx \frac{\theta_{j+1}-\theta_{j-1}}{2h},\quad
\theta''(\xi_{j})\approx \frac{\theta_{j+1}-2\theta_{j}+\theta_{j-1}}{h^{2}}.
$$
代入方程得到非线性代数方程组:
$$
\frac{\theta_{j+1}-2\theta_{j}+\theta_{j-1}}{h^{2}}
+\frac{2}{\xi_{j}}\frac{\theta_{j+1}-\theta_{j-1}}{2h}
+\theta_{j}^{n}=0.
$$
在边界处理上，中心条件$\theta(0)=1$直接给出$\theta_{0}=1$，而$\theta'(0)=0$可通过对称性理解为中心处的偶函数条件。实际离散时可使用Taylor展开给出第一个内部点的近似，也可以引入虚点$\theta_{-1}=\theta_{1}$来处理原点附近的导数条件。外边界可根据研究目标选择：若只计算到给定$\xi_{\max}$，则可令$\theta_{N}$由初值法或渐近估计给定；若研究恒星表面位置，则需要寻找第一个零点$\xi_{1}$满足
$$
\theta(\xi_{1})=0.
$$

由于差分方程中含有非线性项$\theta_{j}^{n}$，需要使用Newton迭代求解。设离散残差为$R(\Theta)=0$，其中
$$
\Theta=(\theta_{1},\theta_{2},\cdots,\theta_{N-1})^{T},
$$
则Newton迭代格式为
$$
J(\Theta^{(m)})\Delta\Theta^{(m)}=-R(\Theta^{(m)}),\quad
\Theta^{(m+1)}=\Theta^{(m)}+\Delta\Theta^{(m)}.
$$
其中Jacobian矩阵$J$由残差对各网格值的偏导数组成。由于每个方程只含相邻三个节点，$J$通常具有三对角结构，可以使用三对角线性方程组算法提高计算效率。

### 2.3 数值实验设计

为了系统比较上述方法，可以从以下几个方面进行数值实验:

1. 不同多变指数$n$的比较。选取$n=0,1,3,5$等典型情形，其中$n=0,1,5$存在解析解，可用于验证算法正确性；$n=3$常用于描述恒星结构中的重要模型。
2. 初始化误差的影响。改变Taylor初始化点$\varepsilon$以及Taylor展开截断阶数，观察中心奇点处理方式对整体误差的影响。
3. 收敛阶检验。对固定步长RK4和有限差分方法分别取不同网格尺度$h$，比较数值解与解析解或高精度参考解之间的误差，验证理论收敛阶。
4. 计算成本比较。记录不同方法在相同误差容许度下的步数、Newton迭代次数、运行时间和最终误差，比较固定步长、自适应步长与有限差分方法的效率差异。
5. 零点位置与物理量计算。对于$n<5$的情形，计算$\theta$首次降为零的位置$\xi_{1}$，并进一步比较相关无量纲质量参数，例如$-\xi_{1}^{2}\theta'(\xi_{1})$的数值误差。
