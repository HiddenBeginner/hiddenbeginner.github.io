---
layout: post
title:  "[MacOS] VS code에서 SSH 서버 연결할 때, Resolver Error 뜰 때 해볼만한 시도 하나"
date:   2022-7-15 20:00
categories: [Python]
use_math: true
comments: true
---

터미널에서는 다음 명령으로 정상적으로 ssh 서버에 연결할 수 있는데, 유독 vs code의 `Remote - SSH`를 사용하여 접속할 때는 원격 서버에 연결할 수 없다는 에러에 봉착했습니다.

~~~bash
ssh [USER]@[HOSTNAME] -p [PORT]
~~~

<br>

"기타 작업"을 누른 후 출력에 보이는 메세지에서 몇몇 오류가 보였습니다. 전체 로그는 개인적인 정보가 많이 담겨 있어서 일부 오류만 적어놓았습니다.
~~~bash
[19:57:19.095] Running script with connection command: ssh -T -D 숫자 -o ConnectTimeout=15 <Host> bash
[19:57:19.275] > /bin/sh: ssh: No such file or directory
[19:57:19.276] Got some output, clearing connection timeout
[19:57:19.523] "install" terminal command done
[19:57:19.523] Install terminal quit with output: /bin/sh: ssh: No such file or directory
[19:57:19.523] Received install output: /bin/sh: ssh: No such file or directory
[19:57:19.523] Failed to parse remote port from server output
[19:57:19.523] Resolver error: Error: 
	at Function.Create (/어쩔/경로)
	at Object.t.handleInstallOutput (/어쩔/경로)
	at Object.t.tryInstall (/어쩔/경로)
	at processTicksAndRejections (/어쩔/경로)
	at async /어쩔/경로
	at async Object.t.withShowDetailsEvent (/어쩔/경로)
	at async Object.t.resolve (/저쩔/경로)
	at async /어쩔/경로
[19:57:19.525] ------
~~~

<br>

여러 시도를 해보았는데도 해결이 되지 않는다면, 저의 글이 해결책이 될 수도 있을 것 같습니다. 우선 `cmd` + `shift` + `p`를 누른 후 **setting**을 검색하여 **기본 설정: 설정 열기(JSON)**에 들어가서 다음과 같은 설정이 있는지 확인해줍니다.
~~~
"terminal.integrated.env.osx": {
        "PATH": ""
    },
~~~

<br>

이 설정이 언제 왜 추가되었을까요? 바로 저의 이전 포스팅 [[MacOS] VS code 터미널에서 파이썬 인터프리터 설정 후에도 다른 경로의 파이썬을 실행하는 경우](https://hiddenbeginner.github.io/python/2022/03/16/vscode_terminal_does_not_point_python_of_virtual_envrionment.html)에서 추가해줬습니다. 하하.

<br>

위의 로그 메세지를 살펴보면 `/bin/sh: ssh: No such file or directory`가 있는데요. `sh`라는 것과 `ssh`의 경로를 못 찾고 있다는 말입니다. 터미널에서 다음을 실행하여 `sh`와 `ssh`가 어디있는지 알아내야 합니다.
~~~bash
whcih sh
which ssh
~~~

<br>

저의 경우 `/bin/sh`와 `/usr/bin/ssh`가 출력되었습니다. 녀석들이 있는 디렉토리를 `"PATH"`에 추가해주면 문제가 해결됩니다. 즉, `settings.json`에서 다음과 같이 수정해줍니다.
~~~
"terminal.integrated.env.osx": {
        "PATH": "/bin:/usr/bin"
    },
~~~

<br>

이렇게 설정하시고 저장하시면, 원격 서버도 잘 접속되고, vs code 터미널에서 알맞은 인터프리터도 잘 찾아내게 됩니다. 이렇게 설정해서 발생하는 에러는 미래의 제가 해결해줄 것입니다.
