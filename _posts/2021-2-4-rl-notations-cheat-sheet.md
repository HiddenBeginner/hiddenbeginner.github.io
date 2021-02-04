---
layout: post
title:  "[강화학습] 강화학습 용어 백과사전"
date:   2021-2-4 21:30
categories: [RL]
use_math: true
comments: true
---

# <center>강화학습 용어 백과사전</center>
**<center>나름대로 해석해 본 강화학습 용어들</center>**<br/><br/>

![figure1](https://raw.githubusercontent.com/HiddenBeginner/hiddenbeginner.github.io/master/static/img/_posts/2021-2-4-rl-notations-cheat-sheet/figure1.png){: width="300" height="300"){: .center}

얼마 전부터 강화학습에 입문하였다. 공부하는 동안 강화학습 용어들이 대체로 추상적이라서 이해하기 어려웠다. 통계 공부할 때 모평균과 표본평균의 차이를 받아들이기 어려웠던 것처럼 말이다. 강화학습을 공부할 때는 모수와 통계량을 잘 구분할 필요가 있다. 강화학습 문제를 기술하기 위한 용어는 대부분 모수라고 생각하면 된다. 

강화학습 이해를 방해하는 다른 한 가지 요소는 용어들의 Recursive한 성질이다. 예를 들어, 현재 시점의 Value 값을 구하기 위해서는 미래 시점의 Value 값을 알아야 한다. 그런데 미래 시점의 Value 값을 구하려고 가보면 더 미래 시점의 Value 값을 가져오라고 한다. 마치 무한츠쿠요미에 빠진 것만 같았다. 이를 처음에 받아들이기 어려웠다. 하지만 일단 이 Recursive한 성질을 받아들이기 시작하면 강화학습의 마법이 펼쳐지기 시작한다. (물론 난 아직 이해 못했다.)

이 포스팅에서 나는 지금까지 이해하기 어려웠던 강화학습 용어들에 내 나름대로의 해석을 덧붙여보았다. 단 한 명에게라도 이해를 돕고 공감을 살 수 있다면 난 만족한다.
<br/><br/>

**들어가기 전에**
- 이 글은 지속적으로 업데이트 됩니다.
- [Richard S. Sutton and Andrew G. Barto, *Reinforcement Learning: an introduction*, 2017](https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf)의 표기법을 최대한 따랐습니다.

---

$s$: State

$a$: Action

$\pi$: Policy (정책)
- Policy는 쉽게 말해서 주어진 상태 (state)에서 어떻게 행동 (action)할지 알려주는 **지침서**이다. 강화학습의 목적은 상황에 맞는 **최적의 지침서**를 찾는 것이다.
- 만약 이 지침서가 주어진 state에서 취해야 할 action을 딱 하나 정해준다면, 이 지침서를 **deterministic policy** 라고 부른다. 그리고 $\pi(s)=a$로 표기한다. 
- 일반적으로, 주어진 state에서 취할 수 있는 action들이 여러 가지이다. 따라서 이 지침서에는 state $s$ 에서 각 action $a$를 취할 확률이 적혀 있다. 즉, 지침서에는 조건부확률 $P(A_t=a \mid S_t=s)$ 들이 적혀 있으며, 이 조건부확률을 간략하게 $\pi(a \mid s)$로 적어준다. 의미는 **주어진 state $s$에서 action $a$를 취할 확률** 이다. 주어진 state에서 취할 action을 조건부확률분포에서 샘플링하여 선택하기 때문에 이 지침서를 **stochastic policy** 라고 부른다.
- 가능한 state와 action 조합이 적을 때는 모든 조합마다 얻을 수 있는 Value function 값을 계산하여 표를 만들어놓는다. 이를 table-lookup 방식이라고 부른다. 메모리에 저장된 표를 보며 action을 선택하는 행위 또한 policy라고 이해할 수 있다.
- 조합의 경우가 너무 많으면 true Policy $\pi$를 구하는 대신 매개변수로 표현되는 모델을 사용한다. $\theta$로 매개변수화된 policy는 $\pi_\theta$ 으로 표기한다. 

$G_t$: Return
- $G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots = \sum_{k=0}^\infty{\gamma^k R_{t+k+1}}$
- $t$ 시점 이후로 Policy를 따라 에피소드를 진행했을 때 얻는 보상들의 합 (시간 경과에 따른 discounted 적용)
- 강화학습은 단순히 "현재 state에서 reward 가장 많이 주는 액션을 취하는 것"이 아니다. 미래의 보상까지 고려하는 "Return 값이 가장 높은 액션을 취하는 것"이다.
    
$V(s)$: State-value function
- $V(s)=\mathbb{E}[G_t \mid S_t=s]$
- 현재 state $s$ 이후로 Policy를 따라 에피소드를 진행했을 때 얻게 되는 return의 기댓값
- state $s$ 에서 시작하여 Policy를 따라가면 1개의 에피소드가 만들어지고 대응하는 $G_t$ 값을 계산할 수 있다. 이 과정을 여러번 반복하면 여러 에피소드와 여러 $G_t$가 만들어질텐데 그것의 평균으로 이해하면 된다. (이 설명은 표본평균에 대한 이야기고, 실제 기대값은 알 수 없는 모수로서 표본평균으로 추정을 한다.)
        
$Q(s,a)$: Action-value function
- $Q(s, a)=\mathbb{E}[G_t \mid S_t=s, A_t=a]$
- 현재 state $s$에서 action $a$를 취한 이후로 Policy를 따라 에피소드를 진행했을 때 얻게 되는 return의 기댓값
- $V(s)$와 마찬가지로 state $s$에서 시작하여 action $a$를 취하고 난 후 Policy를 따라가면 1개의 에피소드와 $G_t$ 값이 완성된다. 이 과정을 여러번 반복하여 얻은 $G_t$들의 평균으로 이해하면 된다.
- state $s$에서 취할 수 있는 action들은 여러가지가 있을 것이다. 각 action $a$에 대한 $Q(s, a)$ 값들을 모두 구해 더하면 $V(s)$ 값이 된다. 즉, 

<div markdown="1">

$$V(s)=\sum_{a \in A}{Q(s, a)}$$
</div>

<details>
<summary>클릭하여 증명 펼치기</summary>
<div markdonw="1">
    
취할 수 있는 action이 n개가 있다고 가정하자. 즉, $A=\left\{a_1, a_2, \cdots, a_N \right\}$.
$$\begin{matrix}
P(G_t|S_t=s)& = &\frac{P(G_t, \;S_t=s)}{P(S_t=s)} & \text{By Bayes' Theorem} \\ 
 & = & \frac{P(G_t, \;S_t=s, \;A_t=a_1)+\cdots+P(G_t, \;S_t=s, \;A_t=a_N)}{P(S_t=s, \;A_t=a_1)+\cdots+P(S_t=s, \;A_t=a_N)} & \text{By Sum Rule} \\
 & = & \frac{P(G_t, \;S_t=s, \;A_t=a_1)}{P(S_t=s, \;A_t=a_1)+\cdots+P(S_t=s, \;A_t=a_N)}+\cdots+\frac{P(G_t, \;S_t=s, \;A_t=a_1)}{P(S_t=s, \;A_t=a_1)+\cdots+P(S_t=s, \;A_t=a_N)} & \text{By 분모나누기} \\
 & = & P(G_t|S_t=s, \;A_t=a_1)+\cdots+P(G_t|S_t=s, \;A_t=a_N) & \text{By Bayes' Theorem} \\
 & = & \sum_{k=1}^{N}P(G_t|S_t=s, A_t=a_k) &&
\end{matrix}$$

Expectation 계산에서 확률부분에 위 성질들 대입하면 쉽게 증명된다.
</div>
</details>

<br/>

$\mu^{\pi}(s)$: Stationary distribution
- Policy $\pi$를 따라 에피소드를 진행했을 때, state $s$가 등장할 확률로 이해하면 쉽다.
- 예를 들어 3가지 state $s_1, s_2, s_3$가 있다고 가정하자. Policy $\pi$를 따라서 에피소드를 진행하다보니 $s_1, s_2, s_3$가 각각 70번, 20번, 10번 등장했다고 하자. 그럼 $\mu^{\pi}(s_1)=\frac{70}{100}=0.7$, $\mu^{\pi}(s_2)=\frac{20}{100}=0.2$, $\mu^{\pi}(s_3)=\frac{10}{100}=0.1$이 된다.
    
$\mathbf{x}(s)$: Feature vector of state $s$
- 사실 state 자체는 구체적이지 않고 모호하다. state는 실수값인가? 혹은 벡터값인가? 그 어떤 설명에서도 "state가 실수이다." 또는 "state는 벡터다."라고 기술하지 않는다. $\mathbf{x}(s)$는 이런 모호한 state 를 구체적인 값으로 나타낼 수 있게 해준다. 예를 들어, 슈퍼마리오 게임에서 특정 state에 대한 feature vector를 만들자면 $[\text{마리오의 좌표}, \text{ 적의 좌표}, \text{ 적과의 거리}, \cdots]$ 가 될 수 있다. feature vector는 다음과 같이 표기할 수 있다.

<div markdown="1">

$$\mathbf{x}(s) = \begin{bmatrix}x_1(s) \\ x_2(s) \\ \vdots \\ x_n(s)\end{bmatrix}$$
</div>

- 각각의 $x_i$ 들은 state로부터 특징을 뽑아내는 함수이다. 예를 들어, 마리오의 좌표를 구해주는 함수, 적의 좌표를 구해주는 함수, 적과의 거리를 구해주는 함수이다.
- state와 $\mathbf{x}(s)$는 엄격하게 구분하지 않고 사용되는 것 같다.
- state $s$에서 가능한 action $a$까지 사용하여 feature vector를 만들 수도 있다. $\mathbf{x}(s, a)$

---

## 참고문헌
사실 이 글을 작성하며 내 지식은 거의 들어가지 않았다. 이 글의 모든 내용은 다음 참고문헌들을 기반으로 작성하였다. 모든 참고문헌 작성자들에게 감사의 인사를 전하고 싶다.

- Video [강화학습의 기초 이론, 팡요랩](https://youtube.com/playlist?list=PLpRS2w0xWHTcTZyyX8LMmtbcMXpd3s4TU)
- Github [https://github.com/seungeunrho/minimalRL](https://github.com/seungeunrho/minimalRL)
- Slide [Introduction to Reinforcement Learning with David Silver](https://www.davidsilver.uk/teaching/)
- Book [Richard S. Sutton and Andrew G. Barto, *Reinforcement Learning: an introduction*, 2017](https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf)
