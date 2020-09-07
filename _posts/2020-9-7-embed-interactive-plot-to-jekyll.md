---
layout: post
title:  "Interactive 시각화 Jekyll 블로그에 업로드하는 쉽고 간단한 방법"
date:   2020-9-7 20:10
categories: [Others,Python]
use_math: true
comments: true
---

# <center>Interactive 시각화 Jekyll 블로그에 업로드하는 쉽고 간단한 방법</center>
이번 포스트에서는 `plotly`와 `bokeh` 등과 같은 interactive한 시각화 결과 그래프를 `Jekyll` 블로그에 업로드하는 방법을 알아본다. 자신의 블로그 환경에 따라서 완전한 해결책이 아닐 수도 있음을 미리 언급하고 넘어가고 싶다. (a.k.a 밑밥)

---

### 1. 시각화 그래프 생성
나는 파이썬에서 주로 `plotly`를 사용하여 interactive한 시각화 그래프를 그린다. 따라서 본 포스트에서는 `plotly`의 [공식문서](https://plotly.com/python/getting-started/)에 있는 그래프를 사용하였다. 하지만 `bokeh`나 `mpld3` 등 시각화 결과를 `.html` 형태로 저장할 수 있다면 모든 괜찮다. 


```python
import plotly.graph_objects as go

fig = go.Figure(data=go.Bar(y=[2, 3, 1]))
fig.show()
```

위 코드를 실행하면 아래와 같은 그래프가 생성된다. 결과 그래프를 단순하게 이미지로 캡처한 것이기 때문에 아직 interactive하지 않다.

![figure1](https://raw.githubusercontent.com/HiddenBeginner/hiddenbeginner.github.io/master/static/img/_posts/2020-9-7-embed-interactive-plot-to-jekyll/figure1.png)

---

### 2. `html` 파일로 저장
두 번 째 단계는 결과 그래프를 `html` 형식의 파일로 저장하는 것이다. 다음 코드를 실행하면 작업하고 있는 폴더에 `example.html`가 만들어질 것이다.


```python
fig.write_html('example.html')
```

---

### 3. `_includes` 폴더에 옮겨 넣기
다음 단계는 위에서 만든 `html` 파일을 `_includes` 폴더에 옮겨 넣는 것이다. 나와 같은 경우에는 `_includes` 폴더에 이미 있는 `html` 파일들과 섞이는 것이 싫어서 `plotly` 폴더를 만들고 그 안에 넣어주었다. 

![figure2](https://raw.githubusercontent.com/HiddenBeginner/hiddenbeginner.github.io/master/static/img/_posts/2020-9-7-embed-interactive-plot-to-jekyll/figure2.png)

---

### 4. `.md` 파일에서 include 해주기
마지막으로 작성 중인 `markdown` 문서에서 원하는 위치에 다음 한 줄을 넣어주면 된다.<br/>
![figure3](https://raw.githubusercontent.com/HiddenBeginner/hiddenbeginner.github.io/master/static/img/_posts/2020-9-7-embed-interactive-plot-to-jekyll/figure3.png)

{% include plotly/example.html %}

---
### 참고한 사이트
[Adding interactive plots to a Jekyll blog](https://www.johnwmillr.com/interactive-plots-in-jekyll/)
