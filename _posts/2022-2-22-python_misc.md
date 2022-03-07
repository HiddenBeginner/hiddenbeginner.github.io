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
- <a href="#private-attribute">비공개 속성 (private attribute)</a>

<br>

---

<span id="init"><span>

### 클래스를 상속 받을 때 `super().__init__()`이 필요한 경우: 부모 클래스의 `__init__` 메서드를 오버라이딩 할 때 
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

    <ipython-input-1-42b3dbe2aae1> in <module>
         36 print(caseA.x)
         37 print(caseB.x)
    ---> 38 print(caseC.x)
    

    AttributeError: 'CaseC' object has no attribute 'x'


<br>

---

<span id="private-attribute"><span>

### 비공개 속성 (Private attribute)
    
누군가가 내가 만든 클래스를 사용할 때 접근하지 못했으면 하는 속성 (attribute)가 있을 수 있다. 속성의 이름 앞에 언더바 2개를 붙이면 비공개 속성을 만들 수 있으며, 비공개 속성은 인스턴스를 통해 접근할 수 없다. 일단은. 아래 코드를 실행해보자 `foo.__private_field`를 실행하면 그런 속성은 없다고 에러 메세지를 보내온다. 이렇게 만든 속성은 자동완성탭에서도 보이지 않는다.


```python
class MyObject:
    def __init__(self):
        self.public_field = 5
        self.__private_field = 10

        
foo = MyObject()
print(foo.public_field)
print(foo.__private_field)
```

    5
    


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    <ipython-input-2-6fcabe9fd5a7> in <module>
          7 foo = MyObject()
          8 print(foo.public_field)
    ----> 9 print(foo.__private_field)
    

    AttributeError: 'MyObject' object has no attribute '__private_field'


사실 이런 비공개 속성은 마음만 먹으면 다 접근할 수 있다. 이름은 비공개 속성이지만 다 알 수 있는 모순적인 친구이다. 클래스의 `__dict__` 속성에 접근해보자. `_MyObject__private_field` 키에 값 10이 있는 것을 확인할 수 있다.


```python
print(foo.__dict__)
```

    {'public_field': 5, '_MyObject__private_field': 10}
    

따라서 다음과 같이 비공개 속성에 접근할 수 있다.


```python
print(foo._MyObject__private_field)
```

    10
    

그럼 도대체 왜 이를 비공개 속성이라고 부를까? 사실 파이썬에서 완전한 비공개 속성을 만드는 것을 굉장히 어렵다고 한다. 이는 파이썬이 만들어진 철학과 관련이 있는데, 코드를 꽁꽁 감춰놓기보다는 공개해서 널리 이롭게 하자 이런 마인드인 것 같다. 비공개 속성을 사용하는 순간 누군가 나의 코드에서 에러가 발생할 때 디버깅하기 어려울 것이다. 다른 이유로는 비공개 속성은 상속이 되지 않는다는 점이 있다. 이 역시 누군가 내 클래스를 상속해서 사용하고 싶을 때, 비공개 속성은 상속 받지 못하여 원활한 코드 사용을 저해할 수 있다.


```python
class MyBaseClass:
    def __init__(self, value):
        self.__value = value
        
    def get_value(self):
        NotImplementedError
    
    
class MyIntegerClass(MyBaseClass):
    def __init__(self, value):
        super().__init__(value)
        
    def get_value(self):
        return int(self.__value)
    
foo = MyIntegerClass(5)
print(foo.get_value())
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    <ipython-input-5-c8aa6443195a> in <module>
         15 
         16 foo = MyIntegerClass(5)
    ---> 17 print(foo.get_value())
    

    <ipython-input-5-c8aa6443195a> in get_value(self)
         12 
         13     def get_value(self):
    ---> 14         return int(self.__value)
         15 
         16 foo = MyIntegerClass(5)
    

    AttributeError: 'MyIntegerClass' object has no attribute '_MyIntegerClass__value'


그 이유는 `__value` 속성은 부모 클래스인 `MyBaseClass`의 `__init__` 메서드에서 정의되었기 때문이다. 마찬가지로 `foo.__dict__`을 출력해보자. `_MyBaseClass__value` 키에 값 5가 저장되어 있는 것을 확인할 수 있다. 따라서 `get_value` 메서드에서 부모 클래스의 `__value`에 접근하고 싶다면, `self._MyBaseClass__value`로 접근해야 할 것이다. 혹여라도 부모 클래스 이름이 바뀌면 자식 클래스에서 호출을 위해 사용한 코드를 모두 수정해야할 것이다. 시간적 여유가 있으면 `self.__value` 대신 `self.value`를 사용했을 경우에는 `foo.__dict__`의 출력값이 어떻게 달라지는지 한번 살펴보길 바란다.


```python
print(foo.__dict__)
```

    {'_MyBaseClass__value': 5}
    

아무튼 비공개 속성은 최대한 사용하지 않는 것이 파이썬 코딩 관습이라고 한다. 나는 자동완성탭에 눈에 보이는 속성들을 줄이기 위하여 비공개 속성을 많이 사용했었는데 이 책을 읽으면서 창피함에 얼굴이 빨개졌다. 앞으로 주의해야겠다. 한편, 코드 제작자가 비공개 속성처럼 다루고 싶은 속성은 앞에 언더바 하나를 쓰는 관습이 있다고 한다 (`self._protect_field`). 앞에 언더바를 포함한 속성은 코딩할 때 클래스 내부적으로만 사용할 목적 (internal API)으로 만든 속성이며 사용자는 굳이 접근하지 않아도 된다는 의미를 포함하고 있다고 한다. 

<br>

---
