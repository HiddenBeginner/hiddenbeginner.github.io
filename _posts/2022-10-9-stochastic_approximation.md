---
layout: post
title:  "[강화학습] Stochastic approximation으로 유도하는 Monte Carlo evaluation과 temporal difference evaluation"
date:   2022-10-9 21:00
categories: [RL]
use_math: true
comments: true
---

![figure1](https://raw.githubusercontent.com/HiddenBeginner/hiddenbeginner.github.io/master/static/img/_posts/2021-2-4-rl-notations-cheat-sheet/figure1.png){: width="300" height="300"){: .center}

<br>

## Stochastic Approximation

어떤 확률 변수 $X$의 기댓값 $\mathbb{E}\left[ X \right]$을 알 수 없을 때, 우리는 주로 표본 평균을 이용하여 실제 평균을 추정한다.

$$
\mathbb{E} \left[ X \right] \approx \frac{1}{N}\sum\limits_{i=1}^{N} X_i
$$

<br>

$X_i$는 우리의 관측 데이터, $N$은 데이터 개수이다. 데이터의 개수가 굉장히 많을 때 모든 $X_i$를 저장하고 있는 것은 비효율적일 수 있다. 특히, 데이터가 추가될 때마다 평균을 구하는 상황에서는 기존 데이터의 덧셈 계산이 중복되기 때문에 위의 방식으로 표본 평균을 계산하는 것은 비효율적이다. 위 식에서 $X_N$만 시그마 밖으로 빼내서 식을 조작해보자.

$$
\begin{matrix}
S_N  & = & \frac{1}{N} \sum\limits_{i=1}^{N} X_i \\
& = & \frac{1}{N} \sum\limits_{i=1}^{N-1} X_i + \frac{1}{N}X_N \\
& = & \frac{N-1}{N} \frac{1}{N-1} \sum\limits_{i=1}^{N-1} X_i + \frac{1}{N}X_N \\
&=&(1-\frac{1}{N})S_{N-1} + \frac{1}{N}X_N \\
& = & S_{N-1} + \frac{1}{N}(X_N-S_{N-1})
\end{matrix}
$$

<br>

위 식은 데이터 $X_N$이 추가되었을 때, 표본 평균 $S_N$을 완전히 다시 계산할 필요 없이 현재 평균 $S\_{N-1}$과 $X_N$ 그리고 $N$을 통해 계산할 수 있다는 것을 보여준다. 이처럼 표본 평균을 구하는 방법을 **incremental mean**이라고 부른다. 어떻게 보면 $X_i$를 샘플링하면서 점점 $\mathbb{E} \left[ X \right]$에 근사시키는 관점에서 **stochastic approximation**으로 부르기도 한다.

<br>

다음 관계를 유심히 기억하면 Monte Carlo와 TD(0)는 물론 TD(1), TD(2), 모두 유도해낼 수 있다.

$$
\mathbb{E} \left[ X \right] \approx S_{N}=S_{N-1} + \frac{1}{N}\left( X_{N} - S_{N-1} \right), \quad \quad (*)
$$

<br>

식 $(*)$에는 크게 세 가지 요소가 있다. 

- $X$    : **Random variable**
- $X_N$ : 샘플링 또는 관측을 통해 실제 값으로 나타난 $X$의 **realization**
- $S_N$  : 기댓값에 대한 추정값 (**Estimate**)

강화학습을 공부할 땐, 항상 수식에서 **random variable**과 **realization**을 잘 구별할 수 있어야 한다.

<br>

---

## Monte Carlo Evaluation

Policy $\pi$에 대한 상태 $s$의 상태 가치 함수가 다음과 같이 정의된다.

$$
v_{\pi}(s)=\mathbb{E} \left[ G_t |S_t =s \right], \quad \quad (**)
$$

<br>

식 $(*)$에 그대로 적용해보자. 

- $X$    :  기댓값 안에 있는 $G_t$는 $X$에 해당한다. (**Random variable**)
- $X_N$ :  $G_t$의 관측값은 $i$ 번째 에피소드에서 상태 $s$에서의 return 값인 $G_{t}^{(i)}$이다. (**Realization**)
- $S_N$  :  이전 상태 가치 함수 추정값 $V_{N-1}(s)$은 $S_{N-1}$에 해당한다. (**Estimate**)

<br>

이를 식 $(*)$에 그대로 대체해서 적어보면 다음과 같이 Monte Carlo evaluation 업데이트 식이 나온다.

$$
V_{N}(s)=V_{N-1}(s) + \frac{1}{N}\left( G_{t}^{(i)} - V_{N-1}(s)\right)
$$

<br>

위 식은 업데이트식이기 때문에 보통 $N$을 제외하고 적어준다. 또한 $\frac{1}{N}$대신 점점 작아지는 작은 값 $\alpha_{N}$을 적는 경우가 많다. 점점 작아지는 상수이어야 수렴성이 증명되지만, 구현에서는 크냥 충분히 작은 상수 하나로 고정하여 사용해도 된다.

$$
V(s)=V(s) + \alpha\left( G_{t}^{(i)} - V(s)\right)
$$

<br>

---

## Temporal Difference Evaluation

식 $(**)$을 변형해보자. 

$$
\begin{matrix}v_\pi(s)&=&\mathbb{E}_\pi[G_t|S_t=s] & \\ &=& \mathbb{E}_\pi[R_{t+1}+\gamma R_{t+2}+\gamma^2R_{t+3}\cdots|S_t=s] & \\ &=& \mathbb{E}_\pi[R_{t+1}+\gamma G_{t+1}|S_t=s] & \\ &=& \mathbb{E}_\pi[R_{t+1}+\gamma\mathbb{E}_\pi[G_{t+1}|S_{t+1}]|S_t=s] & \\ &=&\mathbb{E}_\pi[R_{t+1}+\gamma v_\pi(S_{t+1})|S_t=s] & (***)  \end{matrix}
$$

<br>

두 번째와 세 번째 등호는 return의 정의인 $G_t=R\_{t+1} + \gamma R\_{t+2} + \gamma^2 R\_{t+3} + \cdots$ 을 사용한 것이고, 네 번째 등호는 [law of total expectation](https://en.wikipedia.org/wiki/Law_of_total_expectation)을 사용한 것이다. 다섯 번째 등호는 상태 가치 함수의 정의를 이용했다. 확률에 익숙하지 않으면 받아들이기 어렵겠지만, 기댓값 안에 있는 대문자들은 모두 random variable이고 조건부에 있는 소문자 $s$만이 값이 정해진 realization이다. 

<div class="note-box" markdown="1">

<p class="note-box-title">Law of total expectation</p>

Law of total expectation은 $\mathbb{E} \left[ X \right] = \mathbb{E} \left[ \mathbb{E} \left[ X | Y \right] \right]$이다. 쉽게 이해하자면, 1학년 학생들의 평균 키 $(X)$는 각 반 $(Y)$ 학생들의 평균 키를 구하고 다시 평균을 내서 구할 수 있다는 것을 나타낸다. 조금 더 어려운 이야기를 해보자면 우변에서 대괄호 안의 기댓값은 $X$에 대한 기댓값이고, 바깥 기댓값은 $Y$에 대한 기댓값이다. 바깥 기댓값이 없다고 생각하면 $\mathbb{E} \left[ X | Y = y \right]$ 등으로 적어줘야 한다. 예를 들어, 1반의 평균 키를 나타낸다. 모든 반에 대한 평균을 내는 것이 바깥 기댓값이다.
</div>

<br>    

무튼 식 $(***)$도 결국 기댓값의 형태로 표현되어 있기 때문에 stochastic approximation을 사용할 수 있다. 

- $X$    :  기댓값 안에 있는 $R\_{t+1} + \gamma v\_{\pi}(S\_{t+1})$이 식 $(*)$의 $X$에 대응한다. (**Random variable**)
- $X\_{N}$ :  $R\_{t+1} + \gamma v\_{\pi}(S\_{t+1})$의 관측값은 $r\_{t+1} + \gamma V\_{k-1}(s\_{t+1})$이다. (**Realization**)
- $S_N$  :  이전 상태 가치 함수 추정값 $V\_{N-1}(s_t)$ (**Estimate**)

<br>

이를 식 $(*)$에 대체해서 적어보면 다음과 같이 TD learning의 업데이트 식이 나온다.

$$
V_N(s_t)=V_{N-1}(s_t) + \frac{1}{N} \left( \left[ r_{t+1} + \gamma V_{N-1}(s_{t+1}) \right] - V_{N-1}(s_t) \right)
$$

<br>

Monte Carlo 때와 마찬가지로 $N$을 생략해주고, $\frac{1}{N}$을 $\alpha$로 적어주면 다음과 같아진다.

$$
V(s_t)=V(s_t) + \alpha \left( \left[ r_{t+1} + \gamma V(s_{t+1}) \right] - V(s_t) \right)
$$

<br>
