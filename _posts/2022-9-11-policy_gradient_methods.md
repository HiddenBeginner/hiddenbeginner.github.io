---
layout: post
title:  "[강화학습] 허구한 날 까먹는 Policy gradient theorem 정리"
date:   2022-9-11 21:00
categories: [RL]
use_math: true
comments: true
---

![figure1](https://raw.githubusercontent.com/HiddenBeginner/hiddenbeginner.github.io/master/static/img/_posts/2021-2-4-rl-notations-cheat-sheet/figure1.png){: width="300" height="300"){: .center}

Sutton, Barto 저자의 Reinforcement Learning: An Introduction의 Chapter 13을 읽고 저의 언어로 정리한 글입니다. 위 책을 공부하시는 분들을 위하여 식을 참조하는 번호들은 최대한 책과 일치시켰습니다. 책의 $13.1$절은 policy-based RL의 필요성과 장점들을 나열하고 있으며 이를 블로그 글 전체적으로 녹여내었기 때문에 따로 정리하지 않았습니다. 그럼, 바로 들어가보도록 하겠습니다.

<br>

---

# <center>Policy Gradient Methods in Barto & Sutton</center>

강화학습에서 정책 (policy)은 주어진 상태에서 어떤 행동을 취할지를 알려주는 일종의 지침서 같은 것이다. 보다 더 일반적으로는, 정책 $\pi$는 주어진 상태 $s \in \mathcal{S}$에서 어떤 행동 $a \in \mathcal{A}$을 선택할 조건부 확률이다. 즉, $\pi(a \| s) = \text{Pr} \left[ A_t = a \| S_t = s\right]$ 이다. 만약 상태의 개수와 행동의 개수가 적다면 사람이 직접 각 $(s, a)$마다 확률을 부여하여 정책을 만들 수 있을 것이다. 하지만, 대부분의 환경은 가능한 상태와 행동의 개수가 굉장히 많으며, 심지어 부여할 수 있는 확률 값도 정말 무수히 많을 것이다. 이런 고생을 덜고자 매개변수화된 함수로 정책을 모델링하여 좋은 정책을 찾는 방법을 **policy-based** 방법이라고 한다. 매개변수를 $\mathbf{\theta} \in \mathbb{R}^{d'}$이라고 하면, 이제 매개변수화된 정책은 다음과 같이 적어줄 수 있다.

$$\pi(a | s, \mathbf{\theta}) = \text{Pr} \left[ A_t = a | S_t =s, \mathbf{\theta}_t=\mathbf{\theta}\right].$$

<br>

매개변수의 값에 따라 정책의 성능이 좋을 수도 있고 나쁠 수도 있을 것이다. 우리의 목표는 좋은 정책을 만드는 매개변수를 찾는 것이다. 그러기 위해선 정책의 성능을 평가하는 성능 지표 (performance measure)가 필요하다. 매개변수에 따라 정책의 성능이 달라지므로 성능 지표는 매개변수 값에 의해 결정된다. 따라서 성능 지표를 매개변수에 대한 함수 $J(\mathbf{\theta})$로 적어준다. 

<br>

우리는 성능 지표를 크게 만들어주는 매개변수를 찾기 위해 매개변수에 대한 성능 지표의 그레디언트를 사용할 것이다. 

$$\mathbf{\theta}_{\text{new}}=\mathbf{\theta_{\text{old}}}+\alpha\widehat{\nabla}_{\mathbf{\theta}}{J(\mathbf{\theta}_{\text{old}})}. \quad \quad (13.1)$$

<br>

실제 그레디언트 $\nabla\_{\mathbf{\theta}} J(\mathbf{\theta}\_{\text{old}})$을 찾을 수 있으면 베스트이지만, 일반적으로는 그레디언트에 대한 stochastic 추정치 $\widehat{\nabla}\_{\mathbf{\theta}}{J(\mathbf{\theta}\_{\text{old}})}$를 사용한다. $\widehat{\nabla}\_{\mathbf{\theta}}{J(\mathbf{\theta}\_{\text{old}})}$의 기댓값이 실제 그레디언트 $\nabla\_{\mathbf{\theta}}{J(\mathbf{\theta}\_{\text{old}})}$에 근사하는 추정량을 사용해야 할 것이다. 이와 같이 그레디언트를 사용하여 좋은 정책을 학습하는 방법을 **policy gradient** 방법이라고 부른다. 

<br>

---

## 1. Policy Gradient Theorem
우리는 정책의 성능을 평가하는 지표 $J(\mathbf{\theta})$의 그레디언트를 사용하여 점점 더 좋은 정책을 찾아나갈 것이다. 그럼, 가장 먼저 성능 지표 $J(\mathbf{\theta})$를 정의해야 한다. 이 성능 지표는 주어진 MDP의 설정에 따라 달라질 수 있다. 성능 지표가 달라지면, 그레디언트도 달라질 것이다. 그럼 우리는 성능 지표를 정의할 때마다 그레디언트를 해석적으로 (analytically, 직접 식을 전개하여 푸는 것을 의미) 계산을 해야 하는가? 정말 다행히도 policy gradient theorem은 다양한 성능 지표에 대해서 그레디언트들이 서로 비례한다는 것을 보였다.

<br>

Policy gradient theorem을 조금 더 쉽게 기술하기 위해 주어진 MDP가 finite state space, finite action space를 갖고 episodic이며, $\gamma=1$라고 가정할 것이다. 하지만, continuous state space, continuous action space, infinite horizon에 대해서도 시그마 $\Sigma$만 인테그랄 $\int$로 바꿔서 기술하면 된다. Episodic 환경에서 가장 자연스러운 정책 평가 지표는 에피소드 동안 받은 보상의 총합의 기댓값일 것이다. 즉, 초기 상태의 가치 함수이다. (환경의 초기 상태는 $s_0$로 정해져 있다고 하자.)

$$J(\mathbf{\theta}):=v_{\pi_{\theta}}(s_0). \quad \quad (13.4)$$

<br>

자, 이제 식 $(13.4)$의 그레디언트를 계산해보자. 사실, 썩 쉬워보이지 않는다. 우선, $J(\mathbf{\theta})$는 정책이 취하는 행동에 따라 달라질 수 있다. 그리고, 정책을 따랐을 때 방문하는 상태들에 따라서도 달라질 수 있다. 그래, 정책은 $\mathbf{\theta}$에 대한 함수니깐 그레디언트를 구할 수 있을 것이다. 하지만 정책이 방문한 상태들의 분포는 정책 뿐만 아니라 환경의 transition 모델에 따라 달라질 수 있기 때문에 그레디언트를 계산하는 것이 만만치 않을 것이다. 파라미터 $\mathbf{\theta}$와 성능 지표 $J(\mathbf{\theta})$의 관계를 도식화해보면 다음과 같을 것이다. 

<br>

정말 다행히도 식 $(13.4)$의 그레디언트를 다음과 같이 쉽게 구할 수 있다는 이론이 **policy gradient theorem**이다. 

$$\nabla_{\mathbf{\theta}} J(\mathbf{\theta}) \propto \sum_s \mu_{\pi_{\mathbf{\theta}}}(s) \sum_{a} q_{\pi_{\mathbf{\theta}}}(s,a) \nabla_{\mathbf{\theta}} \pi_{\theta}(a|s),
\quad \quad (13.5)$$

<br>

여기서 $\mu_{\pi_{\theta}}(s)$는 정책 $\pi_{\theta}$를 따랐을 때 상태 $s$에 머무를 확률로 이해하면 된다. Stationary distribution과 비슷하다고 이해하면 되는데, 지금은 episodic MDP를 가정하기 때문에 에피소드를 진행하면서 평균적으로 상태 $s$에 얼마만큼의 비율 동안 머무르는지를 나타낸다. 그리고 이를 on-policy distribution이라고 부른다. Policy gradient theorem에 대한 증명은 천천히 그렇지만 신속하게 추가하도록 하겠습니다.

<br>

식 $(13.5)$는 여전히 복잡해 보이지만, 우려와 다르게 방문한 상태들의 분포 $\mu_{\pi_{\theta}}(s)$ 를 미분하는 일은 발생하지 않았다. $\mu_{\pi_{\theta}}(s)$는 에피소드를 굉장히 많이 진행해보는 방식으로 얼추 구할 수 있을테니깐 말이다. 책에는 나와있지 않지만 식 $(13.5)$를 다음과 같이도 나타낼 수 있다.

$$
\begin{matrix}
\nabla_{\mathbf{\theta}} J(\mathbf{\theta}) & \propto & \sum_s \mu_{\pi_{\mathbf{\theta}}}(s) \sum_{a} q_{\pi_{\mathbf{\theta}}}(s,a) \nabla_{\mathbf{\theta}} \pi_{\mathbf{\theta}}(a|s) & \quad (13.5)\\
& = & \sum_s \mu_{\pi_{\mathbf{\theta}}}(s) \sum_{a} \pi(a|s) q_{\pi_{\mathbf{\theta}}}(s,a) \frac{\nabla_{\mathbf{\theta}} \pi_{\mathbf{\theta}}(a|s)}{\pi_{\mathbf{\theta}}(a|s)}  & \quad (a)\\

& = & \sum_s \mu_{\pi_{\mathbf{\theta}}}(s) \sum_{a} \pi(a|s) q_{\pi_{\mathbf{\theta}}}(s,a) \nabla_{\mathbf{\theta}} \ln \pi_{\mathbf{\theta}}(a|s)  & \quad (b) \\

& = & \mathbb{E}_{\pi_{\mathbf{\theta}}} \left[  q_{\pi_{\mathbf{\theta}}}(S_t, A_t) \nabla_{\mathbf{\theta}} \ln \pi_{\mathbf{\theta}}(A_t|S_t) \right]  & \quad (c)
\end{matrix}
$$

<br>

$(a)$은 그냥 $\frac{\pi_{\mathbf{\theta}}(a\|s)}{\pi_{\mathbf{\theta}}(a\|s)}$를 곱해주고 위치만 바꾼 것이다. $(b)$는 $\frac{d}{dx} \ln f(x)=\frac{f'(x)}{f(x)}$임을 사용한 것이다. 
마지막으로 $(c)$는 $\mathbb{E}\left[ X \right] = \sum_{x}x\;p(x)$임을 사용한 것인데, 확률 $p(x)$에 해당하는 부분은 $\mu_{\pi_{\mathbf{\theta}}}(s)\pi_{\mathbf{\theta}}(a\|s)$이고, 확률변수 $X$에 해당하는 부분이 $q_{\pi_{\mathbf{\theta}}}(S_t,A_t) \nabla \ln \pi_{\mathbf{\theta}}(A_t\|S_t)$이다. 확률변수 (random variable)은 대문자, 결과 (outcome)은 소문자로 표기해주었다. 식 $(c)$처럼 적어주면 좋은 이유는, 실제 기댓값은 구하기 어렵겠지만, 에피소드를 많이 반복하여  $q_{\pi_{\mathbf{\theta}}}(s,a) \nabla \ln \pi_{\mathbf{\theta}}(a\|s)$를 얻고 표본 평균을 내어 실제 기댓값에 근사할 수 있다는 것이다. 그리고 식 $(c)으$로 보는 것이 이후 REINFORCE나 Actor-Crtic 알고리즘을 설명할 때 더 용이하다. 

<br>

---

## 2. REINFORCE: Monte Carlo Policy Gradient
지금까지 policy gradient theorem을 짧게 정리해보자면,
1. 정책을 매개변수된 함수로 모델링하자 $\rightarrow \pi_{\theta}$
2. 정책의 상태가치함수를 목적함수로 정의하자 $\rightarrow J(\theta):= v_{\pi_\theta} (s)$
3. 목적함수의 그레디언트는 다음과 같다
$$\rightarrow \nabla_{\mathbf{\theta}} J(\theta) \propto \mathbb{E}_{\pi_{\mathbf{\theta}}} \left[  q_{\pi_{\mathbf{\theta}}}(S_t, A_t) \nabla_{\mathbf{\theta}} \ln \pi_{\mathbf{\theta}}(A_t|S_t) \right]. \quad \quad (13.6)$$

<br>

즉, 실제 그레디언트가 식 $(13.6)$ 우변의 기댓값에 비례한다는 것이다. 하지만 위의 기댓값을 구하기는 불가능하다. 기댓값 안에 있는 확률변수 $S_t$와 $A_t$는 각각 환경의 전이함수와 정책을 따르는데, 우리는 환경의 전이함수를 모르기 때문이다. 대신 우리는 정책 $\pi_\theta$를 사용하여 환경과 상호작용하며 데이터 $(s_t, a_t)$을 얻는다. 각 데이터 $(s_t, a_t)$에 대응하는 $q_{\pi_{\mathbf{\theta}}}(s_t, a_t)$와 $\nabla \ln \pi_{\mathbf{\theta}}(a_t\|s_t)$를 구할 수 있다면, 우리는 Monte Carlo의 철학을 사용하여 실제 그레디언트를 표본 평균으로 근사시킬 수 있다. 하지만 우리는 행동가치함수를 구하는 것이 쉽지 않다는 것을 알고 있다. 따라서 행동가치함수의 정의를 사용하여 식 $(13.6)$을 정리하면 다음과 같다.

$$\nabla_{\mathbf{\theta}} J(\theta) \propto \mathbb{E}_{\pi_{\mathbf{\theta}}} \left[  G_t \nabla_{\mathbf{\theta}} \ln \pi_{\mathbf{\theta}}(A_t|S_t) \right], \quad \quad (13.7)$$

<br>

이때, return $G_t=R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3}  +\cdots + \gamma^{T - t - 1} R_{T}$이다. 식 $(13.7)$의 관계를 이용하여 stochastic gradient 또는 표본평균을 실제 그레디언트의 근사값으로 사용하여 파라미터를 업데이트하는 알고리즘을 REINFORCE라고 한다. 즉, REINFORCE는 다음과 같은 파라미터 업데이트식을 사용한다.

$$\mathbf{\theta}_{\text{new}}=\mathbf{\theta}_{\text{old}} + \alpha G_t \nabla_{\mathbf{\theta}} \ln \pi_{\mathbf{\theta}_{\text{old}}}(a_t | s_t). \quad \quad (13.8)$$

<br>

책에 나와 있는 REINFORCE의 수도코드는 다음과 같다. 수도코드의 마지막줄 업데이트식은 식 $(13.8)$과 $\gamma^{t}$가 곱해져 있다는 점이 살짝 다르다. 이는 식 $(13.8)$을 유도할 때까지 우리는 non-discounted 상황
($\gamma=1$)을 가정했다. 반면, 아래 수도코드는 일반적으로 discount factor가 있는 경우의 수도코드를 보여준다.

![png](https://raw.githubusercontent.com/HiddenBeginner/hiddenbeginner.github.io/master/static/img/_posts/2022-9-11-policy_gradient_methods/reinforce.png)

<br>

### REINFORCE 장점
책에 따르면 REINFORCE는 좋은 이론적 수렴 성질을 갖는다고 한다. 표본평균은 기댓값의 unbiased estimator라는 Monte Carlo 성질에 의하여 한 에피소드동안 샘플들에서 계산한 그레디언트의 평균은 실제 성능 지표인 식 $(13.4)$의 그레디언트와 동일한 방향을 갖는다. 따라서, 학습률 $\alpha$가 충분히 작다는 가정 아래에서 REINFORCE의 gradient ascent는 실제 성능 지표의 향상을 만들어낸다. 그리고 여느 stochastic gradient ascent처럼 학습률 $\alpha$가 점점 0으로 감소하면, 성능 지표가 local optimum으로 수렴하게 된다.

<br>

### REINFORCE 단점
하지만, 강화학습에서 Monte Carlo 방법론을 사용할 때 추정량의 분산이 크다는 단점이 있었던 것처럼 REINFORCE 역시 그레디언트 추정량의 분산이 크기 때문에 좋은 파라미터에 수렴하기까지 오랜 시간이 걸릴 수 있다.

<br>

---

## 3. REINFORCE with Baseline
강화학습에서는 그레디언트 추정량의 분산을 줄이기 위해 베이스라인 (baseline) 방법을 많이 사용한다. 분산이 큰 이유는 몇몇 있겠지만, 가장 단순한 이유는 기댓값 안의 확률변수가 갖는 값의 크기가 크기 때문이다. 베이스라인 방법을 사용하면 기댓값을 바꾸지 않으면서 확률변수가 갖는 값의 크기를 줄여줄 수 있다.

$$
\nabla_{\mathbf{\theta}} J(\mathbf{\theta}) \propto \sum_s \mu_{\pi_{\mathbf{\theta}}}(s) \sum_{a} \left( q_{\pi_{\mathbf{\theta}}}(s,a) - b(s) \right) \nabla_{\mathbf{\theta}} \pi_{\mathbf{\theta}}(a|s),\quad \quad (13.10)
$$

<br>

여기서 $b(s)$ 행동에 의존적이지 않은 함수가 될 수 있으며, 보통 상태가치함수 $v_{\pi}(s)$이 많이 선택된다. $b(s)$를 빼줬는데도 여전히 비례 관계가 성립하는 이유는 다음과 같이 추가된 항들이 결국 0이기 때문이다.

$$
\begin{matrix}
\sum\limits_a b(s) \nabla_{\mathbf{\theta}}\pi_{\mathbf{\theta}}(a|s) & =&  b(s) \sum\limits_a \nabla_{\mathbf{\theta}}\pi_{\mathbf{\theta}}(a|s) \\
& = & b(s) \nabla_{\mathbf{\theta}} \sum\limits_a \pi_{\mathbf{\theta}}(a|s) \\
& = & b(s) \nabla_{\mathbf{\theta}}1 \\
& = & 0 \\
\end{matrix} \quad .
$$

<br>

이를 이용하여 $G_t$ 대신 $G_t  - b(s_t)$를 업데이트식에 사용하는 방법을 REINFORCE with baseline이라고 한다.

$$\mathbf{\theta}_{\text{new}}=\mathbf{\theta}_{\text{old}} + \alpha \left(  G_t - b(s_t)\right) \nabla_{\mathbf{\theta}} \ln \pi_{\mathbf{\theta}_{\text{old}}}(a_t | s_t). \quad \quad (13.11)$$

<br>

잠시 언급한 것처럼 베이스라인으로 상태가치함수를 많이 사용한다. 하지만 상태가치함수를 직접 계산하기 어렵기 때문에 매개변수를 사용하여 상태가치함수를 모델링하게 된다. 즉, 식 $(13.11)$에서 $b(s_t)$ 대신 $\hat{v}_{\mathbf{w}}(s_t)$로 적어줄 수 있다.  이때 $\mathbf{w} \in \mathbb{R}^{m}$은 상태가치함수 모델의 매개변수이다. 이제 상태가치함수 모델의 매개변수도 업데이트해줘야 하는데 상태가치함수의 정의를 사용하여 업데이트하게 된다. 다음은 책에 나와있는 REINFORCE with Baseline의 수도코드이다.

![png](https://raw.githubusercontent.com/HiddenBeginner/hiddenbeginner.github.io/master/static/img/_posts/2022-9-11-policy_gradient_methods/reinforce-with-baseline.png)

<br>

여기서 주의할 점은 베이스라인을 사용할 경우 정책 $\pi_{\mathbf{\theta}}$와 상태가치함수 $\hat{v}\_{\mathbf{w}}$ 둘 다 학습하게 되는데, 그렇다고 이를 actor-critic 방법이라고 부르지는 않는다. 상태가치함수 $\hat{v}\_{\mathbf{w}}$가 그냥 단순히 베이스라인으로만 사용되고, 비평가 (critic)으로 사용되지 않기 때문이다. <span style="color: red">쉽게 이해하기 위한 설명 추가 필요</span>

<br>

---

## 4. Actor-Critic Methods
REINFORCE는 실제 그레디언트를 추정하기 위해 Monte Carlo 방법론을 사용하였다. Monte Carlo 방법은 unbiased 하지만 분산이 커서 좋은 정책으로 수렴하기까지 오랜 시간이 필요하다. 한편, REINFORCE 업데이트식에는 return $G_t$를 포함하고 있기 때문에 반드시 에피소드가 종료된 후에 파라미터를 업데이트시킬 수 있다. 이로 인해 매 스탭마다 정책을 업데이트할 수 없으며, 특히 continuing 문제에는 적용할 수 없다는 단점이 있다. 한편, temporal difference (TD) 방법론은 biased 하지만 매 스탭마다 업데이트할 수 있다는 장점이 있다. Actor-critic 방법론은 policy gradient 방법론에 TD 방법론을 적용할 수 있게 만들어준다.

<br>

One-step actor-critic은 식 $(13.11)$의 $G_t$ 대신 $r\_{t+1} + \gamma \hat{v}\_{\mathbf{w}}(s\_{t+1})$을 사용한다. 즉,

$$
\begin{matrix}
\mathbf{\theta}_{\text{new}} & =& \mathbf{\theta}_{\text{old}} + \alpha \left( r_{t+1} + \gamma \hat{v}_{\mathbf{w}}(s_{t+1}) - \hat{v}_{\mathbf{w}}(s_t)\right) \nabla_{\mathbf{\theta}} \ln \pi_{\mathbf{\theta}_{\text{old}}}(a_t | s_t)  & \quad (13.13) \\ 
& = & \mathbf{\theta}_{\text{old}} + \alpha \delta_t \nabla_{\mathbf{\theta}} \ln \pi_{\mathbf{\theta}_{\text{old}}}(a_t | s_t) & \quad (13.14)
\end{matrix}
$$

<br>

한편, $\mathbf{w}$도 $G_t$를 사용해서 업데이트하는 대신 $\delta_t$를 사용해서 업데이트하게 된다. Actor-critic 알고리즘은 아래와 같다. 교재에서는 one-step actor-critic 뿐만 아니라 보다 일반적인 n-step까지 다루고 있는데, 이 글에서는 생략하는 편이 좋을 것 같다.

![png](https://raw.githubusercontent.com/HiddenBeginner/hiddenbeginner.github.io/master/static/img/_posts/2022-9-11-policy_gradient_methods/actor-critic.png)

<br>

## 참고문헌
[1] Sutton, R. S., Barto, A. G. (2018). Reinforcement Learning: An Introduction. The MIT Press.
