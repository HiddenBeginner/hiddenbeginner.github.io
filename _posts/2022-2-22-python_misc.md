---
layout: post
title:  "[Python] 파이썬 잡학사전"
date:   2022-2-22 21:00
categories: [Python]
use_math: true
comments: true
---

지금까지 인공지능 및 데이터 과학 분야를 공부하면서 생각보다 많은 파이썬 지식이 필요하지는 않았다. 하지만, 잘 만들어진 깃헙 레포지토리의 코드를 가져와서 내 연구에 맞게 코드를 수정하는 일이 늘어나면서 깊이 있는 파이썬 공부가 필요하다고 느꼈다. 이제 논문을 작성하면 내 코드가 만천하에 공개될텐데, 내 코드를 읽는 사람이 나를 친절한 연구자 또는 인간 세계에 내려온 천사라고 여길 정도로 코드를 짜고 싶다. 지금은 [Effective Python 2nd](https://effectivepython.com/) 책을 읽으며 그때 그때 유익한 정보들을 찾아 정리하고 있다. 

<br>

목차
- <a href="#init">클래스를 상속 받을 때 `super().__init__()`이 필요한 경우</a>


<br>

---

<span id="init"><span>

#### 클래스를 상속 받을 때 `super().__init__()`이 필요한 경우: 부모 클래스의 `__init__` 메서드를 오버라이딩 할 때 
- [https://stackoverflow.com/questions/60015319/is-it-necessary-to-call-super-init-explicitly-in-python](https://stackoverflow.com/questions/60015319/is-it-necessary-to-call-super-init-explicitly-in-python)


```python
class Parent:
    def __init__(self):
        self.x = "{}는 Attribute x를 갖고 있습니다.".format(self.__class__.__name__)
      
    
class CaseA(Parent):
    """
    부모 클래스의 __init__ 메서드를 오버라이딩 하지 않는 경우
    """
    def no_init(self):
        return None
    
    
class CaseB(Parent):
    """
    부모 클래스의 __init__ 메서드를 오버라이딩할 때 super().__init__()을 사용을 사용한 경우
    """
    def __init__(self):
        super().__init__()
        
        
class CaseC(Parent):
    """
    부모 클래스의 __init__ 메서드를 오버라이딩할 때 super().__init__()을 사용을 사용하지 않는 경우
    """
    def __init__(self):
        return None
        
        
parent = Parent()
caseA = CaseA()
caseB = CaseB()
caseC = CaseC()

print(parent.x)
print(caseA.x)
print(caseB.x)
print(caseC.x)
```

    Parent는 Attribute x를 갖고 있습니다.
    CaseA는 Attribute x를 갖고 있습니다.
    CaseB는 Attribute x를 갖고 있습니다.



    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    Input In [6], in <module>
         36 print(caseA.x)
         37 print(caseB.x)
    ---> 38 print(caseC.x)


    AttributeError: 'CaseC' object has no attribute 'x'

