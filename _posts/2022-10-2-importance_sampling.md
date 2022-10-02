---
layout: post
title:  "[강화학습] Importance sampling이란?"
date:   2022-10-2 18:00
categories: [RL]
use_math: true
comments: true
---

![figure1](https://raw.githubusercontent.com/HiddenBeginner/hiddenbeginner.github.io/master/static/img/_posts/2021-2-4-rl-notations-cheat-sheet/figure1.png){: width="300" height="300"){: .center}

<br>

## Importance sampling이란?

Importance sampling은 어떤 확률분포 $p$의 기댓값을 계산하고 싶지만 갖고 있는 데이터가 확률분포 $q$에서 샘플링되었을 때 사용하는 기법이다. 원리는 간단하다.

$$
\begin{matrix}
\mathbb{E}_{X \sim p} \left[ X\right] & = & \sum\limits_{x}xp(x) & \quad (a) \\ 
& = & \sum\limits_{x}x \frac{p(x)}{q(x)} q(x) & \quad (b) \\ 
& = & \mathbb{E}_{X \sim q} \left[ \frac{p(X)}{q(X)} X\right]. & \quad (c) \\
& \approx & \frac{1}{N} \sum\limits_{i=1}^{N} \frac{p(x_i)}{q(x_i)}x_i & \quad (d) 
\end{matrix}
$$

<br>

등식 $(a)$는 기댓값의 정의이고, 등식 $(b)$는 $1=\frac{q(x)}{q(x)}$를 곱해주고 정리해준 것이다. 등식 $(c)$는 다시 기댓값의 정의를 사용한 것이다. 등식 $(4)$는 표본평균으로 기댓값을 근사시키는 것을 나타내며, 이때 $x_i \sim q$이다. 즉, 확률분포 $p$를 따랐을 때 $X$의 기댓값을 구하는 문제가 확률분포 $q$를 따랐을 때 $\frac{p(X)}{q(X)}X$의 기댓값을 구하는 문제로 바뀌었다. 이때, $\frac{p(X)}{q(X)}$를 importance sampling ratio라고 부른다. 

<br>


위 식을 통해 언제 importance sampling을 사용할 수 있는지 알 수 있다. 
1. 확률밀도함수 $p(x)$의 식은 알고 있지만, 모든 $x$를 고려할 수 없어서 샘플 기반 Monte Carlo 방법으로 기댓값을 계산해야 할 때, 그렇지만 우리가 갖고 있는 샘플이 다른 확률분포 $q$에서 샘플링 되었을 때,
2. 확률밀도함수 $p(x)$는 계산할 수 있지만, 확률분포 $p$에서 데이터를 샘플링하기 어려울 때,
3. 분자가 0이 되면 안 되기 때문에 $p(x) \ne 0$인 $x$에 대해서는 $q(x) \ne 0$을 만족할 때.

<br>

---

## 강화학습에서의 importance sampling

Importance sampling은 강화학습에서 다음과 같은 이유로 많이 사용된다.
1. Off-policy 알고리즘에서 행동 정책 (behavior policy) $b(a \| s)$로 환경과 상호작용하여 경험 데이터를 수집하지만, 다른 타겟 정책 (target policy) $\pi(a \| s)$의 가치함수를 계산하고 싶을 때,
2. 현재 정책 $\pi\_{\text{old}}$으로 수집한 경험 데이터를 사용하여 정책을 업데이트 할 때, 업데이트된 정책 $\pi$의 경험 데이터를 따로 모으지 않고 성능지표를 계산하고 싶을 때. 

<br>

1번의 경우, Monte-Carlo control이나 SARSA의 off-policy 버전을 공부할 때 접할 수 있는 내용이다. 2번의 경우 TRPO와 PPO의 목적함수인 surrogate function에 등장한다. TRPO의 목적함수는 다음과 같다.

$$
\operatorname*{maximize}_\theta \hat{\mathbb{E}}_t \left[ \frac{\pi_{\theta}\left( a_t |s_t\right)}{\pi_{\theta_{\text{old}}}\left( a_t |s_t\right)} A_{\theta_{\text{old}}}(s_t, a_t) \right], \quad \quad (1)
$$

$$
\text{subject to} \quad \hat{\mathbb{E}}_t \left[ \operatorname{KL}\left[ \pi_{\theta_{\text{old}}}\left(\cdot|s_t \right), \pi_\theta \left(a_t | s_t \right) \right] \right] \le \delta. \quad \quad (2)
$$

<br>

식 $(1)$에서 $\frac{\pi\_{\theta}\left( a_t \| s_t\right)}{\pi\_{\theta\_{\text{old}}}\left( a_t \| s_t \right)}$가 importance sampling ratio이다. 원래 계산하고 싶은 것은 $\pi_{\theta}$에서 수집한 경험 데이터에 대한 $A\_{\theta\_{\text{old}}}(s_t, a_t)$의 기댓값이다. 이 기댓값은 $\pi\_{\theta}$의 성능와 $\pi\_{\theta\_{\text{old}}}$의 성능 차이 정도로 이해하면 좋다. 성능 차이를 가장 크게 만들어주는 방향으로 $\theta$를 업데이트 해주고 싶은 것이다. 하지만 모든 $\pi\_{\theta}$들마다 경험 데이터를 쌓고 $A\_{\theta\_{\text{old}}}(s_t, a_t)$을 계산하여 그 중 가장 좋은 $\pi\_{\theta}$를 구하는 것을 굉장히 비효율적이다. 따라서, $\pi\_{\theta\_{\text{old}}}$으로부터 수집한 경험 데이터 $(s_t, a_t)$를 사용하되 importance sampling을 도입한 것이다. 

<br>

TRPO와 PPO 논문만 읽었을 때는 "$\pi\_{\theta\_{\text{old}}}$가 현재 정책이고, $\theta$를 업데이트하기 전까지는 $\pi\_{\theta\_{\text{old}}} = \pi\_{\theta}$일텐데 그러면 importance sampling ratio가 1이잖아"라는 생각에 사로잡혀 있었다. 하지만, 논문의 수도 코드와 구현체를 읽어보니 의문이 해소되었다. 현재 정책으로 얻은 경험 데이터를 사용하여 정책을 순차적으로 여러번 업데이트를 하고 있었다.
- 현재 정책 $\pi\_{\theta\_{\text{old}}}$ 으로 정해진 횟수만큼 환경과 상호작용하여 데이터 $\mathcal{D}$ 수집
- 수집한 $\mathcal{D}$으로 정책을 $K$번 업데이트할 예정
- 업데이트에 들어가기에 앞서 수집한 $\mathcal{D}$에 대해서 $\pi\_{\theta\_{\text{old}}}(a_t \| s_t)$ 및 $A\_{\theta\_{\text{old}}}(s_t, a_t)$ 계산 및 이 값은 고정된다.
- 첫 번째 업데이트의 경우 $\theta\_{\text{old}}$가 곧 $\theta$라서 importance ratio는 모두 1이고, 첫 번째 업데이트가 이뤄진 이후부터는 $\theta\_{\text{old}} \ne \theta$가 되는 것

<br>

---

## 글을 마치며

Importance sampling ratio + Monte-Carlo 방법은 원래 구하고 싶은 기댓값에 대한 unbiased estimator이다. 하지만 여느 Monte-Carlo 방법론과 같이 큰 분산을 갖는 estimator라고 한다. 하지만, imporatnce sampling ratio를 곱하는 방식을 적절하게 수정하여 분산을 줄여줄 수 있다고 한다. 또는 Monte-Carlo의 표본평균에서 단순히 표분의 크기로 나눠주는 것이 아니라 importance sampling ration에 대한 가중평균을 취해주면, biased estimator가 되는 대신 통계량의 분산이 줄일 수 있다고 한다. 이런 디테일들을 많이 알고 있다면 강화학습 알고리즘 구현할 때 성능을 조금이라도 끌어올릴 수 있지 않을까 싶었다. Importance sampling의 분산에 대한 이야기는 <a href="#ref1">[1]</a>의 5.5부터 5.9까지 잘 설명하고 있다 (물론 나는 읽어보지 않았다).

<br>

---

## 참고문헌
<p id="ref1">
[1] Sutton, R. S., Barto, A. G. (2018). Reinforcement Learning: An Introduction. The MIT Press.
</p>
