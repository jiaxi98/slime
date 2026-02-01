# Hierarchical Token-Level Baselines for Off-Policy LLM Policy Gradient  
*(完整推导 + 方差表达式证明 + 实现大纲)*

> 本文档给出一种 **hierarchical baseline（分层/递归 baseline）** 的完整推导：  
> 从最原始的 LLM RL 目标与 off-policy policy gradient 出发，逐层引入每一层 baseline $\beta_t^{(s)}$（对固定 $t$，索引 $s\in\{t,t+1,\dots,L\}$ 或只取 $s\in\{t+1,\dots,L\}$），并严格推导最终的 **gradient estimator 方差表达式**。  
> 同时补齐一个在 note 中常被省略的细节：原始 policy gradient 是 $\sum_{t=1}^L$ 的求和，而“总方差等于各项方差之和”需要额外假设（下文会明确写出并解释）。

---

## 0. 记号与设定（LLM RL 通用框架）

- 输入（prompt）$x$，输出（response）序列  
  $$
  y = (y_1,\dots,y_L)
  $$
  为了推导简洁，先假设长度固定为 $L$。若实际长度可变，可以把 EOS 当作终止动作并在实际长度处截断求和（推导基本不变）。

- **目标策略（target policy）**：$\pi_\theta(\cdot\mid x)$  
- **行为策略（behavior / rollout policy）**：$\pi_b(\cdot\mid x)$  
  通常 $\pi_b$ 是旧策略 / actor，用来采样生成 $y$。

- **自回归分解**：
  $$
  \pi_\theta(y\mid x)=\prod_{t=1}^L \pi_\theta(y_t\mid x, y_{<t}),\qquad
  \log\pi_\theta(y\mid x)=\sum_{t=1}^L \log\pi_\theta(y_t\mid x, y_{<t})
  $$
  其中 $y_{<t}=(y_1,\dots,y_{t-1})$。

- **回报（reward / return）**：本文默认使用 **序列级标量回报** $R(x,y)$。  
  若你有 token-level reward $r_t(x,y_{\le t})$，可令 $R=\sum_t r_t$（推导同样成立，只是记号更繁）。

- **优化目标**（对固定 $x$）：
  $$
  J(\theta;x) \;=\; \mathbb E_{y\sim \pi_\theta(\cdot\mid x)}[R(x,y)].
  $$
  若还有 prompt 分布 $x\sim\mathcal D$，则总体目标是 $\mathbb E_{x\sim\mathcal D}J(\theta;x)$，外层期望可在最后再加回去。

---

## 1. 从最原始目标推到 off-policy policy gradient（序列级）

从定义出发：
$$
\nabla_\theta J(\theta;x)
= \nabla_\theta \sum_y \pi_\theta(y\mid x) R(x,y)
= \sum_y \pi_\theta(y\mid x)\,R(x,y)\,\nabla_\theta\log\pi_\theta(y\mid x)
$$
因此 **on-policy** 形式为：
$$
\nabla_\theta J(\theta;x)
=\mathbb E_{y\sim\pi_\theta(\cdot\mid x)}\big[R(x,y)\nabla_\theta\log\pi_\theta(y\mid x)\big].
$$

**off-policy** 采样时我们用 $\pi_b$ 生成 $y$。令序列重要性权重（sequence importance ratio）：
$$
\rho(y) \;=\; \frac{\pi_\theta(y\mid x)}{\pi_b(y\mid x)}.
$$
则有恒等式（support 覆盖时成立）：
$$
\mathbb E_{y\sim\pi_\theta}[f(y)] = \mathbb E_{y\sim\pi_b}\left[\rho(y)f(y)\right].
$$
于是：
$$
\nabla_\theta J(\theta;x)
=\mathbb E_{y\sim\pi_b(\cdot\mid x)}\big[\rho(y)\,R(x,y)\,\nabla_\theta\log\pi_\theta(y\mid x)\big].
$$

把 logprob 分解到 token 级：
$$
\nabla_\theta\log\pi_\theta(y\mid x)
=\sum_{t=1}^L \nabla_\theta\log\pi_\theta(y_t\mid x,y_{<t}).
$$
定义 **score 向量**：
$$
s_t(y)\;:=\;\nabla_\theta\log\pi_\theta(y_t\mid x,y_{<t})\in\mathbb R^d,
$$
则：
$$
\nabla_\theta J(\theta;x)
=\sum_{t=1}^L \mathbb E_{y\sim\pi_b}\big[\rho(y)\,R(x,y)\,s_t(y)\big].
$$
这就是最原始的“序列 IS + token score”形式。

---

## 2. Seq-IS 的关键分解：前缀与后缀重要性权重

对任意 $t$，定义：

- **前缀 ratio**
  $$
  \rho_{\le t}(y)
  :=\frac{\pi_\theta(y_{\le t}\mid x)}{\pi_b(y_{\le t}\mid x)}
  \quad\text{其中 }y_{\le t}=(y_1,\dots,y_t).
  $$

- **后缀 ratio（从 $t+1$ 开始）**
  $$
  \rho_{>t}(y)
  :=\frac{\pi_\theta(y_{>t}\mid x,y_{\le t})}{\pi_b(y_{>t}\mid x,y_{\le t})},
  \quad y_{>t}=(y_{t+1},\dots,y_L).
  $$

则显然：
$$
\rho(y) = \rho_{\le t}(y)\,\rho_{>t}(y).
$$

把它代回每个时间步的梯度项，得到：
$$
g_t(\theta;x)
:=\mathbb E_{y\sim\pi_b}\big[\rho(y)\,R\,s_t\big]
=\mathbb E_{y\sim\pi_b}\big[\rho_{\le t}\,s_t\cdot \rho_{>t}\,R\big].
$$
（这里的“$\cdot$”表示标量乘向量。）

> **直觉**：$\rho_{>t}R$ 可以视为“把未来轨迹用 IS 重标定后的等价回报”，可以用于 $t$ 时刻的 policy gradient 更新。

---

## 3. Hierarchical baseline：逐层引入 $\beta_t^{(s)}$

### 3.1 目标：构造一族无偏、方差更小的 $g_t$ 估计器

我们希望构造一个随机梯度估计器：
$$
\hat g_t(y)
=\rho_{\le t}(y)\;s_t(y)\;\widehat A_t(y),
\quad y\sim \pi_b(\cdot\mid x),
$$
满足：
1) **无偏性**：$\mathbb E_{\pi_b}[\hat g_t]=g_t$  
2) **方差可控/可最小化**：$\mathrm{Var}(\hat g_t)$ 能写成易处理的形式

其中 $\widehat A_t$ 是一个标量（类似“advantage/return”），我们将用分层 baseline 来构造它。

---

### 3.2 分层系数 $\alpha_t^{(s)}$ 的定义（后缀局部 ratio）

对固定 $t$，对任意 $s>t$ 定义：
$$
\alpha_t^{(s)}(y)
:=\frac{\pi_\theta(y_{t+1:s}\mid x, y_{\le t})}{\pi_b(y_{t+1:s}\mid x, y_{\le t})}.
$$
并约定：
$$
\alpha_t^{(t)}(y):=1,\qquad
\alpha_t^{(R)}(y):=\alpha_t^{(L)}(y):=\rho_{>t}(y)=\frac{\pi_\theta(y_{t+1:L}\mid x,y_{\le t})}{\pi_b(y_{t+1:L}\mid x,y_{\le t})}.
$$

注意：$\alpha_t^{(s)}$ 是**从 $t+1$ 到 $s$** 的“部分后缀”IS ratio；而 $\alpha_t^{(R)}$（或 $\alpha_t^{(L)}$）是从 $t+1$ 到 $L$ 的完整后缀 ratio。

---

### 3.3 Baseline $\beta_t^{(s)}$ 的索引范围：$s\in\{t,\dots,L\}$ vs $s\in\{t+1,\dots,L\}$

- **最一般**的写法允许 $s=t$：$\beta_t^{(t)}$ 相当于“常规状态 baseline”（不乘未来 ratio）。
- 如果你希望严格满足你提到的 “$t<s\le L$”（即**不包含** $s=t$ 这一层），只需令
  $$
  \beta_t^{(t)} \equiv 0
  \quad\text{并把求和从 }s=t+1\text{ 开始}.
  $$
  推导结构完全一致。

下文为了统一，先写成 $s=t,\dots,L$ 的一般形式；你可以随时把 $s=t$ 那项删掉。

---

### 3.4 分层 baseline 的最终形式（展开式）

我们构造：
$$
\widehat A_t(y)
=\alpha_t^{(R)}(y)\,R(x,y) \;-\; \sum_{s=t}^{L}\alpha_t^{(s)}(y)\,\beta_t^{(s)}.
\tag{HB}
$$
于是分层 baseline 的 token 梯度估计器为：
$$
\boxed{
\hat g_t(y)
=\rho_{\le t}(y)\;s_t(y)\;
\left(
\alpha_t^{(R)}(y)\,R(x,y) \;-\; \sum_{s=t}^{L}\alpha_t^{(s)}(y)\,\beta_t^{(s)}
\right).
}
\tag{E}
$$

接下来我们要做两件最关键的证明：

1) 证明 **无偏性**：$\mathbb E_{\pi_b}[\hat g_t]=g_t$。  
2) 推导 **方差表达式**：$\mathrm{Var}(\hat g_t)=\mathbb E[\rho_{\le t}^2\|s_t\|^2(\cdots)^2]-C_t$。

---

## 4. 无偏性证明（最细致版本）

无偏性要证明：
$$
\mathbb E_{\pi_b}[\hat g_t]
=\mathbb E_{\pi_b}\big[\rho_{\le t}s_t\alpha_t^{(R)}R\big]
-
\sum_{s=t}^L \mathbb E_{\pi_b}\big[\rho_{\le t}s_t\alpha_t^{(s)}\beta_t^{(s)}\big]
=g_t.
$$
其中第一项正是 $g_t$（见第 2 节）。所以只需证明：

> **Lemma 1（baseline 项期望为 0）**：对任意 $s\in\{t,\dots,L\}$，若 $\beta_t^{(s)}$ 只依赖于 $x$ 与 $y_{<t}$（即不依赖当前动作 $y_t$），则
> $$
> \mathbb E_{y\sim\pi_b}\big[\rho_{\le t}(y)\,s_t(y)\,\alpha_t^{(s)}(y)\,\beta_t^{(s)}\big]=0.
> $$

---

### 4.1 两个小恒等式（都是关键）

**(i) score 的零均值性质（在目标策略下）**  
对任何给定 $(x,y_{<t})$：
$$
\mathbb E_{y_t\sim\pi_\theta(\cdot\mid x,y_{<t})}\big[\nabla_\theta\log\pi_\theta(y_t\mid x,y_{<t})\big]
=\sum_{y_t}\pi_\theta(y_t\mid\cdot)\nabla_\theta\log\pi_\theta(y_t\mid\cdot)
=\sum_{y_t}\nabla_\theta\pi_\theta(y_t\mid\cdot)
=\nabla_\theta 1
=0.
\tag{S0}
$$

**(ii) IS ratio 的“条件期望为 1”**  
对任何给定 $(x,y_{\le t})$ 与任意 $s>t$：
$$
\mathbb E_{y_{t+1:s}\sim\pi_b(\cdot\mid x,y_{\le t})}\big[\alpha_t^{(s)}(y)\mid x,y_{\le t}\big]
=\sum_{y_{t+1:s}}
\pi_b(y_{t+1:s}\mid x,y_{\le t})\,
\frac{\pi_\theta(y_{t+1:s}\mid x,y_{\le t})}{\pi_b(y_{t+1:s}\mid x,y_{\le t})}
=\sum_{y_{t+1:s}} \pi_\theta(y_{t+1:s}\mid x,y_{\le t})
=1.
\tag{IS1}
$$
当 $s=t$ 时，$\alpha_t^{(t)}\equiv 1$，也显然成立。

---

### 4.2 Lemma 1 的严格证明（一步不跳）

从要证的期望开始，使用全期望分解（tower property）：

$$
\mathbb E_{\pi_b}\big[\rho_{\le t}s_t\alpha_t^{(s)}\beta_t^{(s)}\big]
=\mathbb E_{y_{<t}\sim\pi_b}\left[
\beta_t^{(s)}(x,y_{<t})\cdot
\mathbb E_{y_t\sim\pi_b(\cdot\mid x,y_{<t})}
\left[
\frac{\pi_\theta(y_{\le t}\mid x)}{\pi_b(y_{\le t}\mid x)}
s_t
\cdot
\mathbb E_{y_{t+1:s}\sim\pi_b(\cdot\mid x,y_{\le t})}[\alpha_t^{(s)}\mid x,y_{\le t}]
\right]
\right].
$$

注意这里我们把 $\beta_t^{(s)}$ 拉到了最外层，因为它不依赖 $y_t$（只依赖 $y_{<t}$）；而 $\alpha_t^{(s)}$ 的期望我们用 (IS1) 处理：

$$
\mathbb E_{y_{t+1:s}}[\alpha_t^{(s)}\mid x,y_{\le t}] = 1.
$$

于是上式化简为：
$$
\mathbb E_{\pi_b}\big[\rho_{\le t}s_t\alpha_t^{(s)}\beta_t^{(s)}\big]
=\mathbb E_{y_{<t}\sim\pi_b}\left[
\beta_t^{(s)}(x,y_{<t})
\cdot
\mathbb E_{y_t\sim\pi_b(\cdot\mid x,y_{<t})}
\left[
\frac{\pi_\theta(y_{\le t}\mid x)}{\pi_b(y_{\le t}\mid x)}\,s_t
\right]
\right].
$$

把前缀 ratio 拆开：
$$
\frac{\pi_\theta(y_{\le t}\mid x)}{\pi_b(y_{\le t}\mid x)}
=\frac{\pi_\theta(y_{<t}\mid x)}{\pi_b(y_{<t}\mid x)}\cdot
\frac{\pi_\theta(y_t\mid x,y_{<t})}{\pi_b(y_t\mid x,y_{<t})}
=\rho_{<t}(y)\cdot \rho_t(y).
$$
因此内层对 $y_t$ 的期望变成：
$$
\mathbb E_{y_t\sim\pi_b}
\left[\rho_{<t}\rho_t s_t\right]
=\rho_{<t}\cdot
\mathbb E_{y_t\sim\pi_b}
\left[
\frac{\pi_\theta(y_t\mid x,y_{<t})}{\pi_b(y_t\mid x,y_{<t})}
\nabla_\theta\log\pi_\theta(y_t\mid x,y_{<t})
\right].
$$

把 $\pi_b$ 抵消后，得到：
$$
\rho_{<t}\cdot
\sum_{y_t}\pi_\theta(y_t\mid x,y_{<t})\nabla_\theta\log\pi_\theta(y_t\mid x,y_{<t})
=\rho_{<t}\cdot 0
=0
$$
其中用到了 (S0)。因此整个外层期望为 0，Lemma 1 证毕。

---

### 4.3 得到无偏性结论

由 Lemma 1：
$$
\mathbb E_{\pi_b}[\hat g_t]
=\mathbb E_{\pi_b}[\rho_{\le t}s_t\alpha_t^{(R)}R]
=g_t,
$$
所以分层 baseline 不改变梯度的期望（无偏）。

---

## 5. 方差推导：从定义到 note 中的“简洁形式”（最细致证明）

### 5.1 我们用的“向量方差（trace variance）”定义

$\hat g_t$ 是一个向量随机变量。常用的标量方差度量是：
$$
\mathrm{Var}(\hat g_t)
:=\mathbb E\big[\|\hat g_t-\mathbb E\hat g_t\|^2\big].
\tag{Vdef}
$$
它等价于协方差矩阵的 trace。

展开平方范数：
$$
\begin{aligned}
\mathbb E\big[\|\hat g_t-\mathbb E\hat g_t\|^2\big]
&=\mathbb E\big[\|\hat g_t\|^2 - 2\hat g_t^\top\mathbb E\hat g_t + \|\mathbb E\hat g_t\|^2\big]\\
&=\mathbb E[\|\hat g_t\|^2] - 2(\mathbb E\hat g_t)^\top(\mathbb E\hat g_t) + \|\mathbb E\hat g_t\|^2\\
&=\mathbb E[\|\hat g_t\|^2] - \|\mathbb E\hat g_t\|^2.
\end{aligned}
\tag{Vexpand}
$$

令
$$
C_t := \|\mathbb E[\hat g_t]\|^2,
$$
则
$$
\boxed{
\mathrm{Var}(\hat g_t) = \mathbb E[\|\hat g_t\|^2] - C_t.
}
\tag{V1}
$$

> 重要：由于第 4 节已证明 $\mathbb E[\hat g_t]=g_t$ 与 $\beta_t^{(s)}$ 无关，  
> 所以 **$C_t$ 与所有 baseline 系数无关**，它是一个常数项。  
> 因此“最小化方差”等价于“最小化二阶矩 $\mathbb E\|\hat g_t\|^2$”。

---

### 5.2 计算二阶矩：为什么会出现那种“干净的”形式

回忆估计器（式 (E)）：
$$
\hat g_t(y)
=\rho_{\le t}(y)\;s_t(y)\;
\Delta_t(y),
$$
其中我们把标量残差记为
$$
\Delta_t(y)
:=\alpha_t^{(R)}(y)\,R(x,y) \;-\; \sum_{s=t}^{L}\alpha_t^{(s)}(y)\,\beta_t^{(s)}.
\tag{Delta}
$$

由于 $\rho_{\le t}$ 与 $\Delta_t$ 都是标量，$s_t$ 是向量，所以范数平方满足：
$$
\|\hat g_t\|^2
=\|\rho_{\le t}\,s_t\,\Delta_t\|^2
=(\rho_{\le t})^2\;\|s_t\|^2\;(\Delta_t)^2.
\tag{norm}
$$
这是唯一用到的线性代数事实：$\|a v\|^2=a^2\|v\|^2$。

对 $y\sim\pi_b$ 取期望：
$$
\mathbb E[\|\hat g_t\|^2]
=\mathbb E_{y\sim\pi_b(\cdot\mid x)}
\left[
\left(\frac{\pi_\theta(y_{\le t}\mid x)}{\pi_b(y_{\le t}\mid x)}\right)^2
\cdot
\left\|\nabla_\theta\log\pi_\theta(y_t\mid x,y_{<t})\right\|^2
\cdot
\left(
\alpha_t^{(R)}R - \sum_{s=t}^{L}\alpha_t^{(s)}\beta_t^{(s)}
\right)^2
\right].
\tag{secondmoment}
$$

代回 (V1)，得到：

$$
\boxed{
\mathrm{Var}(\hat g_t)
=\mathbb E_{\pi_b}
\left[
\left(\rho_{\le t}\right)^2 \|s_t\|^2
\left(
\alpha_t^{(R)}R - \sum_{s=t}^{L}\alpha_t^{(s)}\beta_t^{(s)}
\right)^2
\right]
-
C_t.
}
\tag{Vt}
$$

这就是 note 中“非常简单的 variance 形式”的严格来源：  
**(Vexpand) 的方差恒等式 + (norm) 的范数缩放性质**。

---

## 6. 总梯度是 $\sum_t$ ：为什么总方差常被写成各项方差之和？

真实的总梯度估计器是：
$$
\hat g(y) = \sum_{t=1}^L \hat g_t(y).
$$

严格地说：
$$
\mathrm{Var}(\hat g)
=\mathrm{Var}\left(\sum_{t=1}^L \hat g_t\right)
=\sum_{t=1}^L \mathrm{Var}(\hat g_t)
+ 2\sum_{1\le t<s\le L}\mathrm{Cov}(\hat g_t,\hat g_s),
\tag{Vsum}
$$
其中协方差（对向量）用 trace 形式等价地写成
$$
\mathrm{Cov}(\hat g_t,\hat g_s)
=\mathbb E\big[\hat g_t^\top \hat g_s\big]
-
\mathbb E[\hat g_t]^\top\mathbb E[\hat g_s].
$$

因此，“总方差=各项方差之和”并不是恒等式；它需要假设交叉协方差为 0（或近似忽略）。

---

### 6.1 我们采用的两个假设（与你照片里的写法对齐）

为了把推导变得可实现/可解释，通常会使用以下两类假设（它们是 **常见近似**，不保证严格成立，但在高维参数与实际训练中往往足够好）：

**Assumption A（跨时间步近似不相关 / 交叉项可忽略）**  
对任意 $t\neq s$，假设
$$
\mathrm{Cov}(\hat g_t,\hat g_s) \approx 0,
\quad\text{即}\quad
\mathbb E[\hat g_t^\top \hat g_s]\approx \mathbb E[\hat g_t]^\top\mathbb E[\hat g_s].
\tag{A}
$$
在许多 LLM 设置里，这个假设常通过“score 向量近似正交”来动机化。

**Assumption B（score 内积的 delta 结构）**（与你图片的形式一致）  
假设存在常数 $C$（或随 $t$ 变化的 $C_t$），使得
$$
\mathbb E\Big[s_t(y)^\top s_s(y)\Big]
=\delta_{ts}\,C
\quad(\text{或 }\delta_{ts}C_t),
\tag{B}
$$
其中 $\delta_{ts}$ 为 Kronecker delta。直观上：不同位置的 score 在期望意义下近似正交，且同一位置的期望平方范数近似为常数。

> 说明：  
> - (B) 是比 (A) 更具体的一种结构化假设。  
> - 从 (B) 推到 (A) 还需要额外“标量权重与 score 方向弱相关”等条件；实践里常把交叉项视为噪声并忽略。

---

### 6.2 在假设下，总方差的近似形式

在 (A) 下：
$$
\boxed{
\mathrm{Var}(\hat g)
\approx \sum_{t=1}^L \mathrm{Var}(\hat g_t).
}
\tag{Vapprox}
$$

把 (Vt) 代入，得到总方差的可优化近似目标：
$$
\boxed{
\mathrm{Var}(\hat g)
\approx
\sum_{t=1}^L
\left\{
\mathbb E_{\pi_b}
\left[
(\rho_{\le t})^2 \|s_t\|^2
\left(
\alpha_t^{(R)}R - \sum_{s=t}^{L}\alpha_t^{(s)}\beta_t^{(s)}
\right)^2
\right]
-
C_t
\right\}.
}
\tag{Vtotal}
$$
由于每个 $C_t$ 都与 $\beta$ 无关，最小化总方差就等价于分别最小化每个 $t$ 的二阶矩项。

---

## 7. 最优 $\beta_t^{(s)}$ 的求解：线性系统 / 加权最小二乘（WLS）

对固定 $t$，在 (Vt) 中忽略常数 $-C_t$，我们等价地最小化：
$$
\min_{\{\beta_t^{(s)}\}_{s=t}^L}
\;
\mathbb E_{\pi_b}\Big[
w_t(y)\cdot
\big(
z_t(y) - \sum_{s=t}^L \phi_{t,s}(y)\beta_t^{(s)}
\big)^2
\Big],
\tag{WLS}
$$
其中定义：

- 权重（来自方差表达式）
  $$
  w_t(y):=(\rho_{\le t}(y))^2\cdot\|s_t(y)\|^2
  $$
- “标签”
  $$
  z_t(y):=\alpha_t^{(R)}(y)\,R(x,y)
  $$
- 特征
  $$
  \phi_{t,s}(y):=\alpha_t^{(s)}(y)
  \quad (s=t,\dots,L).
  $$

这就是一个标准的 **加权线性回归 / 加权最小二乘** 问题。

对任意 $s\in\{t,\dots,L\}$，对 $\beta_t^{(s)}$ 求偏导并令 0：

$$
\frac{\partial}{\partial \beta_t^{(s)}}
\mathbb E\Big[w_t(z_t-\sum_r \phi_{t,r}\beta_t^{(r)})^2\Big]
=\mathbb E\Big[ -2 w_t \phi_{t,s}\,(z_t-\sum_r \phi_{t,r}\beta_t^{(r)})\Big]
=0.
$$

等价于 **正规方程（normal equations）**：
$$
\mathbb E\Big[w_t\phi_{t,s}\,z_t\Big]
=\sum_{r=t}^L \mathbb E\Big[w_t\phi_{t,s}\phi_{t,r}\Big]\beta_t^{(r)}.
\tag{NE}
$$

写成矩阵形式更清晰：

- 向量 $\beta_t\in\mathbb R^{(L-t+1)}$，分量是 $\beta_t^{(t)},\dots,\beta_t^{(L)}$
- 矩阵 $A_t\in\mathbb R^{(L-t+1)\times(L-t+1)}$：
  $$
  (A_t)_{sr} = \mathbb E[w_t\phi_{t,s}\phi_{t,r}]
  $$
- 向量 $b_t\in\mathbb R^{(L-t+1)}$：
  $$
  (b_t)_s = \mathbb E[w_t\phi_{t,s}z_t]
  $$

则：
$$
\boxed{A_t\,\beta_t=b_t.}
\tag{LinSys}
$$

> 实践里：  
> - 直接求逆很贵且数值不稳；  
> - note 建议用 **Gauss–Seidel** 这类递归迭代法在线更新；  
> - 或仅保留少数层 $s$（如只用 $s=L$）得到更便宜的近似（例如 Seq-IS）。

---

## 8. 在一般 LLM RL 框架下的实现大纲（工程落地）

下面给一个可直接落地到 PPO/REINFORCE 风格训练的实现大纲。你可以把它理解为“如何在现有 RLHF pipeline 里替换/增强 advantage 估计”。

---

### 8.1 输入/输出与数据流

**输入：**
- prompts batch $\{x_i\}_{i=1}^B$
- 行为策略 $\pi_b$（生成时用）
- 当前策略 $\pi_\theta$（训练时更新）
- reward 计算器：RM 或 verifiable reward，输出 $R(x_i,y_i)$

**输出：**
- 一个用于反向传播的 loss（其梯度等于你的估计 policy gradient）

---

### 8.2 Rollout（采样）

对每个 $x_i$，用行为策略采样生成：
$$
y_i\sim \pi_b(\cdot\mid x_i).
$$
记录（至少）：
- 每步行为 logprob：$\log \pi_b(y_{i,t}\mid x_i,y_{i,<t})$
- 生成的 token 序列（含 EOS）
- 序列级 reward：$R_i=R(x_i,y_i)$

---

### 8.3 重新打分（re-score）得到 $\log \pi_\theta$

对每个样本序列，用当前策略 teacher-forcing 计算：
$$
\log \pi_\theta(y_{i,t}\mid x_i,y_{i,<t}).
$$

并计算每步 log-ratio：
$$
d_{i,t}
:= \log \pi_\theta(y_{i,t}\mid\cdot) - \log \pi_b(y_{i,t}\mid\cdot).
$$

用前缀和快速得到各种 ratio（数值上推荐全在 log-space 做累加）：

- 前缀 log-ratio：
  $$
  D_{i,t}=\sum_{k=1}^t d_{i,k}
  \quad\Rightarrow\quad
  \rho_{\le t}^{(i)}=\exp(D_{i,t})
  $$
- 任意后缀片段 ratio：
  $$
  \alpha_{t}^{(s)}(i)=\exp(D_{i,s}-D_{i,t}),\quad s>t,
  \qquad \alpha_t^{(t)}(i)=1,
  $$
- 完整后缀：
  $$
  \alpha_t^{(R)}(i)=\alpha_t^{(L)}(i)=\exp(D_{i,L}-D_{i,t}).
  $$

> **稳定化建议**：  
> 重要性权重在长序列会爆炸，实践中常需要 clip：  
> $\rho_{\le t}\leftarrow \mathrm{clip}(\rho_{\le t}, \rho_{\min},\rho_{\max})$，  
> 或在 log-space clip $D_{i,t}$。

---

### 8.4 估计/学习 $\beta_t^{(s)}$（三种可选路线）

#### 路线 1：mini-batch 上直接估计线性系统（WLS）
对固定 $t$，用 batch 样本近似期望：

- 权重（理论最优）：
  $$
  w_{i,t}=(\rho_{\le t}^{(i)})^2\cdot \|s_{i,t}\|^2
  $$
  其中 $s_{i,t}=\nabla_\theta\log\pi_\theta(y_{i,t}\mid\cdot)$。  
  **注意：$\|s_{i,t}\|^2$ 在大模型上很贵**，实践常用近似：
  - 直接忽略它：$w_{i,t}\approx(\rho_{\le t}^{(i)})^2$
  - 或用便宜 proxy（例如只看最后一层/LoRA 子参数的 per-example grad norm）

- 特征：$\phi_{i,t,s}=\alpha_t^{(s)}(i)$
- 标签：$z_{i,t}=\alpha_t^{(R)}(i)\,R_i$

构造经验矩阵与向量：
$$
\hat A_{t,sr}=\sum_{i=1}^B w_{i,t}\,\phi_{i,t,s}\phi_{i,t,r},\qquad
\hat b_{t,s}=\sum_{i=1}^B w_{i,t}\,\phi_{i,t,s}z_{i,t}.
$$
加 ridge 稳定：
$$
\beta_t \leftarrow (\hat A_t+\lambda I)^{-1}\hat b_t.
$$
复杂度随 $(L-t+1)^3$ 增长，长序列不划算 → 建议只保留少数层 $s$。

#### 路线 2：Gauss–Seidel（递归迭代求解）
把线性系统 $A_t\beta_t=b_t$ 当作固定点方程，循环更新每个 $\beta_t^{(s)}$：
$$
\beta_t^{(s)} \leftarrow \frac{\mathbb E[w_t\phi_{t,s}(z_t-\sum_{r\ne s}\phi_{t,r}\beta_t^{(r)})]}{\mathbb E[w_t\phi_{t,s}^2]}.
$$
用 batch 估计期望即可。优点：在线、稳定、不需要求逆。

#### 路线 3：用一个 baseline 网络 amortize（最像 value model）
训练一个网络 $b_\psi(x,y_{<t},t,s)\approx \beta_t^{(s)}$。  
用加权回归目标拟合 (WLS)：
$$
\min_\psi\ \mathbb E\big[w_t(z_t-\sum_s \phi_{t,s}b_\psi(\cdot))^2\big]
$$
或在你只保留少数层 $s$ 时变成普通加权 MSE。  
优点：推理快、无需解方程；缺点：要训练额外模型（但通常比 critic 更简单）。

---

### 8.5 构造优势/残差并做策略更新

对每个样本与时间步，计算分层残差（对应式 (HB)）：
$$
\Delta_{i,t}
=\alpha_t^{(R)}(i)\,R_i
-
\sum_{s=t}^L \alpha_t^{(s)}(i)\,\beta_t^{(s)}.
$$

得到每步权重：
$$
w^{\text{PG}}_{i,t}
=\rho_{\le t}^{(i)}\cdot \Delta_{i,t}.
$$

**实现 policy gradient 的一个关键点**：  
如果你直接写 loss = $-w^{PG}\cdot \log\pi_\theta$，要记得 **stop-gradient** 处理权重（因为推导里的梯度已经是 score function 形式）：

$$
\mathcal L(\theta)
=-\sum_{i,t}\mathrm{stopgrad}\big(w^{PG}_{i,t}\big)\cdot \log\pi_\theta(y_{i,t}\mid x_i,y_{i,<t}).
$$

这样反向传播得到的梯度就是：
$$
\nabla_\theta \mathcal L
=-\sum_{i,t} w^{PG}_{i,t}\; \nabla_\theta \log\pi_\theta(y_{i,t}\mid\cdot),
$$
与我们推导的 estimator 形式一致。

---

### 8.6 与 PPO / KL 正则的结合方式

如果你使用 PPO（或 KL penalty），通常会有额外的项：
- KL($\pi_\theta\|\pi_{\text{ref}}$) penalty
- PPO ratio clip

你可以把这里的 $\Delta_{i,t}$ 当作 advantage / return 的替代，把它送进 PPO 的 surrogate。  
但注意：PPO 的目标本身会对 ratio 求导，它不再是纯 score-function estimator，所以要仔细检查“是否仍无偏”。工程上更常见的做法是：

- 用这套 hierarchical baseline 来**构造更低方差的 advantage**；
- 在 PPO 里仍按 PPO 的方式用 ratio clip 更新（此时 estimator 变成“PPO 近似”）。

---

## 9. 复杂度与实践建议（最关键的落地取舍）

1) **不要真的用全层 $s=t,t+1,\dots,L$**：复杂度 $O(L^2)$ 甚至更高。  
   推荐只选少数层，例如：
   - 只用 $s=L$：退化为 Seq-IS baseline
   - 用 $s\in\{t,L\}$：两层 baseline（note 后几页就是这种分析）
   - 用分块边界：比如每隔 $K$ 个 token 一个层

2) **重要性权重一定要稳定化**：clip / truncate / self-normalize，否则方差可能更大。

3) **$\|s_t\|^2$ 权重往往会被近似掉**：理论最优需要它，但工程常取 $w_t\approx\rho_{\le t}^2$ 或 $w_t\approx 1$。

4) **把 $\beta$ 当回归问题**：很多时候训练一个小 baseline 网络比解线性系统更稳、更省。

---

## 10. 小结（你得到的最终“最重要公式”）

- 分层 baseline 的 token 梯度估计器：
  $$
  \hat g_t
  =\rho_{\le t}s_t
  \left(
  \alpha_t^{(R)}R - \sum_{s=t}^L \alpha_t^{(s)}\beta_t^{(s)}
  \right)
  $$
- 其方差（trace variance）：
  $$
  \mathrm{Var}(\hat g_t)
  =\mathbb E_{\pi_b}
  \left[
  (\rho_{\le t})^2\|s_t\|^2
  \left(
  \alpha_t^{(R)}R - \sum_{s=t}^L \alpha_t^{(s)}\beta_t^{(s)}
  \right)^2
  \right]
  -C_t
  $$
- 在交叉协方差可忽略的假设下：
  $$
  \mathrm{Var}\Big(\sum_t \hat g_t\Big)\approx \sum_t \mathrm{Var}(\hat g_t).
  $$
- 对固定 $t$ 的最优 $\beta_t$ 满足加权最小二乘线性系统 $A_t\beta_t=b_t$。

---

## 参考
- 你提供的笔记：*Off-Policy Variance Reduction with Token-level Baselines*（内部 note）
