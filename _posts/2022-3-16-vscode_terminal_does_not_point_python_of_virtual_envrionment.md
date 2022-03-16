---
layout: post
title:  "[MacOS] VS code 터미널에서 파이썬 인터프리터 설정 후에도 다른 경로의 파이썬을 실행하는 경우"
date:   2022-3-16 14:00
categories: [Python]
use_math: true
comments: true
---


Mac OS에서만 발생하는 문제인지는 모르겠지만, Visual Studio Code (VS code)의 터미널에서 파이썬 인터프리터를 잘 설정해주고, 심지어 가상환경 활성화까지 했는데도 설치된 패키지들이 임포트되지 않는 문제가 발생했다. 예를 들어 VS Code 터미널에서

~~~bash
> conda activate venv
> (venv) python -m "import tensorflow as tf"
~~~


라고 명령했을 때, 텐서플로우가 없다는 에러가 발생하는 문제이다. venv 가상환경에는 분명히 텐서플로우가 설치되었는데도 이와 같은 문제가 발생하는 이유는 다음과 같다. VS Code 터미널에서 다음을 명령해보자.

~~~bash
> (venv) which python
~~~


또는

~~~bash
> (venv) which python3
~~~


아마 경로의 끝부분이 `venv/bin/python`와 같이 가상환경 이름의 갖는 디렉토리의 파이썬을 사용하지 않고, `usr/bin/python` 이 나올 것이다. 가상환경을 활성화하면 가상환경의 파이썬을 사용해야 하는데, 원래부터 설치되어 있던 파이썬을 사용하는 것이다. (내가 아는 선에서는 Mac OS에는 파이썬이 기본적으로 설치되어 있다. 해당 파이썬을 지칭하고 있는 것 같다.)<br><br>

해결책은 간단하다. VS Code에서 `Cmd` + `Shfit` + `P`를 누르고 setting을 검색하여 `Preferences: Open settings (JSON)`에 들어가자. 그럼 여러 설정 인자들이 이미 적혀있을텐데, 다음의 것도 추가해주면 된다. 이전 항목 뒤에 쉼표를 쓰는 것을 잊지 말자 !!

~~~json
"terminal.integrated.env.osx": {
        "PATH": ""
}
~~~

<br>


출처: [https://stackoverflow.com/a/55043991](https://stackoverflow.com/a/55043991)

관련된 github issue: [https://github.com/Microsoft/vscode-python/issues/4434#issuecomment-466600591](https://github.com/Microsoft/vscode-python/issues/4434#issuecomment-466600591)

---


