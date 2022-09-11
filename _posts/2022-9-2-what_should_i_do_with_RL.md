---
layout: post
title:  "나의 강화학습 회고록 (부제: 왜 강화학습은 현실에서 잘 사용되지 않는가?)"
date:   2022-9-2 22:00
categories: [RL, Others]
use_math: true
comments: true
---

![figure1](https://raw.githubusercontent.com/HiddenBeginner/hiddenbeginner.github.io/master/static/img/_posts/2021-2-4-rl-notations-cheat-sheet/figure1.png){: width="300" height="300"){: .center}

내가 왜 지금 강화학습을 공부하고 있는지, 강화학습과 전혀 무관한 연구실에서 왜 강화학습을 하고 싶어하는지, 내가 진짜 강화학습 분야로 성공할 수 있을지, 보다 더 객관적으로 생각해보기 위해 이 블로그글을 남깁니다. 

(a.k.a 주간 블로깅 챌린지 채우기용)

<br>

## 내가 강화학습을 공부하기 시작한 "진짜" 이유 (회고)

나는 왜 강화학습을 하고 싶을까? 많은 인공지능 분야 중 왜 하필 강화학습을 선택했을까? 

요새 유행하는 MBTI 용어를 빌려 나에 대해 설명하자면 나는 N (직관형)이 아닌 S (감각형)이다. 상상력이 부족해서 그런지 추상적인 개념이나 설명은 아무리 읽어도 잘 머릿속에 남지 않는다. 추상적인 개념을 이해하기 위해서는 반드시 눈에 보이는 예시가 필요하거나, 조금 더 잘 와닿는 개념들을 사용하여 논리적으로 기술되어 있어야 한다.

<div class="note-box" markdown="1">

<p class="note-box-title">잠깐 !</p>

참고로 나의 MBTI 유형은 ISFP이다.
</div>

<br>

아, 왜 갑자기 이런 이야기를 하냐면, 이런 나의 뇌의 작동 방식 때문에 나는 인공지능이 무엇인지, 인간의 지능이 무엇인지, 인간이 어떻게 학습하는지 등에 대해 나의 견해나 철학이 따로 없다. 지도학습/비지도학습에 비해 강화학습이 실제 인간이 학습하는 방식에 가깝다는 의견도 있지만, 그래봤자 결국 강화학습도 최적화 (optimization) 문제를 푸는 것에 지나지 않는다고 생각한다. "강화학습이 진짜 인공지능에 한 걸음 다가갈 수 있는 방법이다." 등의 믿음이 있어서 강화학습을 선택한 것이 아니다.

<br>

따라서 나는 어떤 철학을 가지고 강화학습 공부를 시작하지 않았다. 강화학습을 선택한 솔직한 이유는 다른 인공지능 분야 보다 더 진입장벽이 높아보였기 때문이다. 다른 분야가 쉽다는 것은 절대 아니고, 다른 분야보다 수학이 조금 더 많이 사용되는 것 같았다. 객관적으로 나는 컴공 전공자분들에 비해 코딩 능력이 현저히 떨어진다. 하지만 학부 때 수학을 전공했기 때문에 강화학습 논문의 무수한 수식 공격에도 당황하지 않을 기본 자세가 되어 있었다. 다른 분야보다 조금 더 전문성을 갖추기 유리했다고 생각했다.

<br>


위의 고민은 결국 내가 지금 어떤 공부를 해야 큰 돈을 많이 벌 수 있을까에 대한 고민으로 귀결된다.
어떤 분야를 선택해야 남들보다 더 큰 전문성을 갖춰서 좋은 조건으로 고용될 수 있을까에 대한 고민.
그래 사실 취업하고 싶어서 강화학습 공부를 한 것이다. 다른 인기 분야에서 내가 경쟁력을 갖추기 어려울 것 같아서 수학이 많은 강화학습을 선택한 것이다.

<br>

## 나 강화학습으로 취업할 수 있을까..? 강화학습이 산업에서 잘 사용되지 않는 이유들
그런데 강화학습을 공부하면 할수록 이 놈을 실제 산업에 사용하기 어렵겠다는 생각이 많이 들었다. 실제로 다른 인공지능 분야에 비해 강화학습 직무를 채용하는 회사를 찾기 어렵다. [노션 페이지](https://dongjinlee.notion.site/06944c4c6f224cd9bc8c58c32d40dffa?v=0e2ae884e5c94881882fdb0e1898709b)에 강화학습 직무를 채용하는 회사들을 정리해 놓았다. 강화학습 직무를 채용하는 회사가 적을 뿐만 아니라, "강화학습 직무"라고 딱 명시된 회사도 몇 없다. 취업해서 돈 벌려고 강화학습을 선택했는데, 취업마저도 잘 안 된다니, 이건 너무한거 아니냐고~~

<br>

내가 생각했을 때, 강화학습이 실제 산업에서 사용되기 어려운 이유는 다음과 같다.

<br>

### 1. 강화학습의 불안정성 in 구현

강화학습 분야는 CNN, RNN, Transformer와 같은 모델 아키텍처를 만드는 분야라기 보다는 순차적 결정 문제 (sequential decision making problem)를 푸는 일련의 "알고리즘"을 만드는 분야이다. 그러다 보니 다른 분야보다 코드가 상대적으로 길다. 긴 코드를 구현하다 보면 사소한 실수가 발생할 수 있다. 이 사소한 코드가 에러를 발생시켜 디버깅할 수 있으면 큰 문제가 되지 않는다. 하지만 사소한 실수가 있음에도 코드가 정상적으로 돌아갈 때가 있다. 이 경우 에이전트의 처참한 성능의 원인이 도대체 어디에서 발생했는지 찾기가 너무 어렵다. "이 문제는 다른 분야에도 있는 것 아니야?"라고 생각이 들 것이다. 하지만 강화학습 분야에서는 이 문제를 더 심각하게 받아들이는 것 같다. 이를 잘 알 수 있는 부분이 강화학습의 대표 패키지인 `Stable Baseline 3` <a href="#ref1">[1]</a>는 "이 코드는 잘 구현되었으니 여기서부터 알고리즘 추가하세요."를 목적으로 만들어진 패키지이다.

<br>

뿐만 아니라 코드 구현에 실수가 없다고 하더라고 한 강화학습 알고리즘 안에 너무 많은 하이퍼파라미터가 있다. 우선 심층강화학습은 딥러닝 분야가 갖는 공통적인 하이퍼파라미터들

- 모델 네트워크에 하이퍼파라미터 (레이어 개수, 노드 수 등)
- 모델 학습에 대한 하이퍼파라미터 (에폭수, 배치 사이즈, 학습률)

<br>

을 갖고 있다. 하나의 알고리즘 안에 여러개의 뉴럴 네트워크가 있을 수 있다. 예를 들면, Soft Actor-Crtic (SAC)에서는 5개의 네트워크를 사용한다 <a href="#ref2">[2]</a>. 뉴럴 네트워크 관련 하이퍼라미터를 제외하고도 강화학습 알고리즘 내에 많은 하이퍼파라미터들이 더 존재한다. 가장 대표적인 것이 할인률 $\gamma$, 소프트 타겟 네트워크 factor $\tau$, 정책의 stochasticity를 조절하는 temperature $\alpha$, TD learning의 $\lambda$ 등이 있다. 필요한 하이퍼파라미터의 종류는 강화학습 알고리즘마다 다르다는 것도 골치아프다.

<br>

그리고 무엇보다 이 하이퍼파라미터의 작은 변화에도 에이전트의 성능이 굉장히 크게 달라지게 된다 <a href="#ref3">[3]</a>. 물론 다른 딥러닝 분야도 하이퍼파라미터 선택에 민감한 것은 마찬가지라고 생각할 수 있다. 하지만 강화학습에서는 상대적으로 더 크게 영향을 받는다. 강화학습이 풀고자 하는 순차적 결정 문제는 한 시점에서 발생한 작은 에러가 다음 시점의 에이전트의 의사결정에 영향을 준다. 초기의 작은 에러가 나비효과를 만들어 엄청 큰 에러를 만들어 낼 수 있다. 강화학습의 지도학습 버전인 behavior cloning이 실패하는 이유이다 <a href="#ref4">[4]</a>.

<br>

이 외에도 강화학습 알고리즘 내 구현 디테일에 따라서도 에이전트의 성능이 크게 변할 수 있다. 
강화학습의 대표 on-policy model free 알고리즘인 PPO는 TRPO의 알고리즘을 단순화하면서도 높은 성능 향상을 보이며 큰 주목을 받았었다. 하지만, PPO의 성능 향상이 알고리즘적 차이에서 온 것이 아닌 사소한 구현 디테일 차이에서 온 것일 수도 있다고 한다 <a href="#ref5">[5]</a>.

<br>

정리하자면, 회사 입장에서는 "우리의 문제가 잘 정의되었는지, 그리고 문제 해결을 위해 강화학습이 적절한지"에 따라 강화학습의 성패가 결정되어야 하는데, 어쩌면 왜 실패했는지도 모른채 실패할 수도 있다는 것이다. 

<br>

### 2. 강화학습의 결과의 큰 분산
1번 내용과 비슷한 이야기로 느껴질 수 있지만, 사소한 차이가 있다. 잘 학습된 에이전트일지라도 환경의 랜덤성 (stochasticity)에 의해 기대에 못 미치는 결과를 만들거나 때때로 돌이킬 수 없는 처참한 결과를 만들어 낼 때가 있다. 지도학습 관점으로 예시를 들자면, 10개의 검증 데이터 중 9개의 검증 데이터에 대해서는 100점을 맞는데 1개의 검증데이터에 대해서는 0점을 맞을 수 있다는 것이다. 이 문제를 잘 정리해 놓은 논문은 아직 찾지 못했다. 그냥 내가 스스로 코드를 돌려보며 느낀 부분이다. 강화학습의 대표격 예제 문제 [CartPole](https://www.gymlibrary.dev/environments/classic_control/cart_pole/)에서도 대부분의 경우에는 195점 이상을 기록하다가도 시작하자마자 픽 쓰러지는 경우가 있었다. 

<br>

### 3. 환경 만들기의 어려움
강화학습은 지도학습/비지도학습과 다르게 데이터를 미리 준비할 필요가 없다는 장점이 있다.
에이전트가 환경과 상호작용하면서 데이터를 만들어내기 때문이다. 환경의 상태 ($s$)를 보고 에이전트는 알맞은 행동 ($a$)을 수행하여 환경의 다음 상태($s'$)와 보상 ($r$)을 관찰한다. 그럼 $(s, a, r, s')$이 하나의 데이터가 되는 것이고, 이 데이터들을 학습에 이용하게 된다.
이건 어디까지나 에이전트가 상호작용할 수 있는 "환경"이 있을 때 이야기이다. 환경은 매 시점 상태를 갖고 있어야 하고, 행동을 받았을 때 어떻게 상태를 바꿀 것인지, 보상을 얼마나 줄 것인지 모두 사람이 하나하나 만들어줘야 한다. 특히, 원하는 테스크를 강화학습이 잘 풀 수 있게 보상을 디자인 해주는 것도 굉장히 어려운 문제이다. 

<br>

주로 강화학습 환경은 시뮬레이션 환경으로 만들어주지만, 현실 세계에 도입하기 위해서는 결국 현실의 환경과 상호작용하면서 데이터를 수집해야 할 것이다. 현실과 상호작용하는 것은 결국 지도학습/비지도학습이 겪는 데이터 수집 비용 문제와도 직결된다. 그래서 최근에는 시뮬레이션 환경에서 학습된 모델을 실제 환경에서도 잘 적용될 수 있도록 하는 Sim2Real 연구가 활발히 진행되고 있다. Sim2Real도 생각보다 잘 작동하지 않아서 사람이 직접 모은 데이터를 사용해서 에이전트를 학습시키는 imitation learning과 학습 및 deployment 단계에서까지 환경과 상호작용이 없다고 가정하는 offline RL이 유행하고 있는 것 같다.

<br>

실제 환경과 직접 상호작용하면서 online learning을 하면 더할 나위 없겠지만 여기에도 문제가 있다. 바로 안정성 문제이다. 학습이 되지 않은 에이전트는 임의의 행동을 취하며 상태와 행동을 탐색 (exploration)한다. 이때 에이전트의 임의의 행동이 환경을 돌이킬 수 없는 상태로 만들어 버릴 수 있다. 또는 에이전트나 환경에 있는 다른 에이전트에게 상해를 입힐 수도 있다. 로봇이 임의 행동을 했다가 자기 관절이 꺾어버리면 고장나면 2,000만 원이 날라갈 수 있다. 자율주행자동차가 아 여기서 좌회전하면 어떤 보상을 받을까 하고 좌회전하다가 인명사고가 발생할 수도 있다. 

<br>

이렇게 탐색 과정 중에서 안전한 행동/상태만 하는 방법을 연구하는 분야로 safe exploration 또는 safety RL 분야가 있다. Safety RL이 조금 더 큰 개념이고, 그 하위 개념으로 safe exploration이 있는 것 같은데, 확실하지는 않다. Gaussian process로 환경을 모델링하여 안전한 상태만을 탐색하는 연구 <a href="#ref6">[6]</a>, <a href="#ref7">[7]</a>, Lyapunov function을 사용해서 환경의 안정성을 보장하는 연구 <a href="#ref7">[7]</a>, <a href="#ref8">[8]</a>, Constrained optimization을 이용하는 연구 <a href="#ref8">[8]</a>, <a href="#ref9">[9]</a>, <a href="#ref10">[10]</a>등이 있다. Safety RL 분야가 수학이 조금 더 빡센 느낌이다. 

<br>

## 그럼 난 어떻게 해야 하는가?
나는 집착이 굉장히 심한 편이다. 내가 한번 강화학습을 선택한 이상 나는 이 강화학습을 손쉽게 놓아 주지 않을 것이다. 강화학습이 실제 세계에 사용되기 어렵다면, 실제 세계에 사용할 수 있게 개선하면 되는 것이다. 위에 내가 적은 이유들 중에서 나는 특히 2번 강화학습 결과의 큰 분산 부분을 다루고 싶다. 

<br>

이미 말했던 것처럼 잘 학습된 에이전트일지라도 환경의 랜덤성에 의해 처참한 결과를 만들어 낼 수 있다. 평상시에는 에이전트가 의사결정을 아주 훌륭하게 수행할지라도 아주 낮은 확률로 돌이킬 수 없는 처참한 결과를 만든다면 실제 공정에 에이전트를 도입할 수 없을 것이다. 로봇이 물건을 잘 만들다가 대뜸 공정을 부숴버릴 수도 있고, 자율주행자동차가 잘 주행하다가 급브레이크를 잡아서 사고를 유발할 수도 있다는 것이다. 알파고도 이세돌 9단의 대국에서 예상 밖의 수를 마주하자 패착을 두기 시작했다. 나는 이런 돌발행동을 하지 않는 에이전트를 만들고 싶다.

<br>

3번에서 소개한 safe reinforcement learning과 비슷한 맥락처럼 느껴지지만 내가 원하는 것은 조금 다르다. 나는 언제 돌려도 일관성 있는 결과를 주는 에이전트를 만들고 싶다. 학습 때 얻었던 성능을 deployment 단계에서도 보장 받고 싶다. 그런데 나와 같이 생각하는 연구는 없는 것 같다. 아니면 내가 키워드를 잘 못 찾는 것 같다. 논문을 찾기가 어려웠다.

<br>

요새 드는 생각은 이 모든 걱정이 인간이 모은 데이터를 이용하면 해결되지 않을까 하는 것이다. 인간이 데이터를 모으면 explicit하게 환경을 만들 필요도 없고, 학습 중 안정성 걱정도 하지 않아도 된다. 요새 유행하는 offline RL을 공부해야 할까? 하지만, offline으로 학습한 모델이더라도 실제 deployment 단계에서 학습 때의 성능을 만든다고 보장할 수 없다. 이는 다시 내가 말한 일관성 있는 결과를 보장는 에이전트를 개발해야 하는 것을 의미한다.

<br>

## 글을 마치며,
지금까지 긴 글을 작성했는데 글의 서문에 적은 질문들
- 왜 지금 강화학습을 공부하고 있는지
- 강화학습과 전혀 무관한 연구실에서 왜 강화학습을 하고 싶어하는지
- 내가 진짜 강화학습 분야로 성공할 수 있을지

에 제대로 된 답변을 내린게 하나도 없다는 것을 깨달았다. 

<br>

내가 독자라면 서문 읽고 대뜸 MBTI 이야기 하더니 갑자기 취업하고 싶다고 하다가 또 취업 안 된다고 찡얼 거리는 곳까지 읽고 글을 넘겼을 것 같다. 지금이라도 글의 제목과 서문을 바꿔야하는 것일까? 하지만, 난 이 바보 같이 일관성 없는 글이 마음에 든다. 글에 일관성은 없지만, 최근 나의 머릿 속에 생각과 고민들을 잘 정리할 수 있는 시간이 되었기 때문이다. 

<br>

여전히 내가 지금 연구실에서 강화학습 공부를 계속 해야 하는지, 그리고 그 끝은 성공일지 예상되지 않는다. 현실적이고 객관적으로 봤을 때 취업에 더 용이한 다른 분야로 돌아가는게 맞는 것 같다. 근데 너무 늦은 감이 있기도 하고, 지금까지 쌓아올린 것도 없기도 해서 두려운 것 같다. 

<br>

마침내 두서도 없고 결론도 없는 글이 되었다. 그래도 행복하다. 주간 블로깅 챌린지를 이어가는게 내 진짜 목적이었으니깐. 하하.

<br>

## 참고문헌
<p id="ref1">
[1] https://github.com/DLR-RM/stable-baselines3 
</p>
<p id="ref2">
[2] Haarnoja, Tuomas, Aurick Zhou, Pieter Abbeel와/과Sergey Levine. “Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor”. In Proceedings of the 35th International Conference on Machine Learning, 편집자： Jennifer Dy와/과Andreas Krause, 80:1861–70. Proceedings of Machine Learning Research. PMLR, 2018. https://proceedings.mlr.press/v80/haarnoja18b.html.
</p>
<p id="ref3">
[3] Henderson, Peter, Riashat Islam, Philip Bachman, Joelle Pineau, Doina Precup, David Meger. “Deep Reinforcement Learning That Matters”. In Proceedings of the Thirty-Second AAAI Conference on Artificial Intelligence and Thirtieth Innovative Applications of Artificial Intelligence Conference and Eighth AAAI Symposium on Educational Advances in Artificial Intelligence. AAAI Press, 2018. 
</p>
<p id="ref4">
[4] Ross, S., Gordon, G.J., & Bagnell, J.A. (2011). A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning. AISTATS.
</p>
<p id="ref5">
[5] Engstrom, Logan, Andrew Ilyas, Shibani Santurkar, Dimitris Tsipras, Firdaus Janoos, Larry Rudolph와/과Aleksander Madry. “Implementation Matters in Deep RL: A Case Study on PPO and TRPO”, 2020. https://openreview.net/forum?id=r1etN1rtPB.
</p>

<p id="ref6">
[6] Wachi, Akifumi, Yanan Sui, Yisong Yue, Masahiro Ono. “Safe Exploration and Optimization of Constrained MDPs Using Gaussian Processes”. Proceedings of the AAAI Conference on Artificial Intelligence 32, 호 1 (2018년 4월 26일). https://doi.org/10.1609/aaai.v32i1.12103.
</p>

<p id="ref7">
[7] Berkenkamp, Felix, Matteo Turchetta, Angela P. Schoellig, Andreas Krause. “Safe Model-Based Reinforcement Learning with Stability Guarantees”. In Proceedings of the 31st International Conference on Neural Information Processing Systems, 908–19. NIPS’17. Red Hook, NY, USA: Curran Associates Inc., 2017.
</p>

<p id="ref8">
[8] Chow, Yinlam, Ofir Nachum, Edgar Duenez-Guzman, Mohammad Ghavamzadeh. “A Lyapunov-based Approach to Safe Reinforcement Learning”. In Advances in Neural Information Processing Systems, Vol 31. Curran Associates, Inc., 2018. https://proceedings.neurips.cc/paper/2018/hash/4fe5149039b52765bde64beb9f674940-Abstract.html.
</p>

<p id="ref9">
[9] Achiam, Joshua, David Held, Aviv Tamar, Pieter Abbeel. “Constrained Policy Optimization”. In Proceedings of the 34th International Conference on Machine Learning, 22–31. PMLR, 2017. https://proceedings.mlr.press/v70/achiam17a.html.
</p>

<p id="ref10">
[10] Achiam, Joshua, Dario Amodei. “Benchmarking Safe Exploration in Deep Reinforcement Learning”, 2019.
</p>
