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
- <a href="#getter-setter">@property와 @property.setter 데코레이터</a>

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


<br>

사실 이런 비공개 속성은 마음만 먹으면 다 접근할 수 있다. 이름은 비공개 속성이지만 다 알 수 있는 모순적인 친구이다. 클래스의 `__dict__` 속성에 접근해보자. `_MyObject__private_field` 키에 값 10이 있는 것을 확인할 수 있다.


```python
print(foo.__dict__)
```

    {'public_field': 5, '_MyObject__private_field': 10}
    

<br>

따라서 다음과 같이 비공개 속성에 접근할 수 있다.


```python
print(foo._MyObject__private_field)
```

    10
    

<br>

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


<br>

그 이유는 `__value` 속성은 부모 클래스인 `MyBaseClass`의 `__init__` 메서드에서 정의되었기 때문이다. 마찬가지로 `foo.__dict__`을 출력해보자. `_MyBaseClass__value` 키에 값 5가 저장되어 있는 것을 확인할 수 있다. 따라서 `get_value` 메서드에서 부모 클래스의 `__value`에 접근하고 싶다면, `self._MyBaseClass__value`로 접근해야 할 것이다. 혹여라도 부모 클래스 이름이 바뀌면 자식 클래스에서 호출을 위해 사용한 코드를 모두 수정해야할 것이다. 시간적 여유가 있으면 `self.__value` 대신 `self.value`를 사용했을 경우에는 `foo.__dict__`의 출력값이 어떻게 달라지는지 한번 살펴보길 바란다.


```python
print(foo.__dict__)
```

    {'_MyBaseClass__value': 5}
    

<br>

아무튼 비공개 속성은 최대한 사용하지 않는 것이 파이썬 코딩 관습이라고 한다. 나는 자동완성탭에 눈에 보이는 속성들을 줄이기 위하여 비공개 속성을 많이 사용했었는데 이 책을 읽으면서 창피함에 얼굴이 빨개졌다. 앞으로 주의해야겠다. 한편, 코드 제작자가 비공개 속성처럼 다루고 싶은 속성은 앞에 언더바 하나를 쓰는 관습이 있다고 한다 (`self._protect_field`). 앞에 언더바를 포함한 속성은 코딩할 때 클래스 내부적으로만 사용할 목적 (internal API)으로 만든 속성이며 사용자는 굳이 접근하지 않아도 된다는 의미를 포함하고 있다고 한다. 

<br>

---

<span id="getter-setter"><span>

### `@property`와 `@property.setter` 데코레이터

파이썬에서는 보통 다음과 같이 객체의 속성 (attribute)에 접근하거나 값을 새롭게 할당한다.


```python
class MyClass:
    def __init__(self, sequence):
        self.sequence = sequence
   

foo = MyClass([1, 2, 3, 4, 5])
# sequence 속성 접근
print("sequence 속성: ", foo.sequence)

# sequence 속성 할당
foo.sequence = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print("sequence 속성: ", foo.sequence)
```

    sequence 속성:  [1, 2, 3, 4, 5]
    sequence 속성:  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    

<br>

때론 객체의 속성에 접근하거나 값을 할당할 때 단순히 데이터를 주고 받는 것을 넘어서 더 많은 것을 수행하고 싶을 수 있다. 예를 들어, 위의 `sequence` 속성에 있는 원소의 개수를 나타내는 `num_elements` 속성을 만들고 싶다고 하자. 생각할 수 있는 가장 단순한 방법은 다음과 같이 `__init__` 함수에서 `num_elements` 속성을 만드는 것이다.


```python
class MyClass:
    def __init__(self, sequence):
        self.sequence = sequence
        self.num_elements = len(self.sequence)
   

foo = MyClass([1, 2, 3, 4, 5])
# sequence 속성 접근
print("num_elements 속성: ", foo.num_elements)

# sequence 속성 할당
foo.sequence = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print("num_elements 속성: ", foo.num_elements)
```

    num_elements 속성:  5
    num_elements 속성:  5
    

<br>

하지만, `sequence`가 새로운 리스트로 바뀌어도 `num_elements` 속성에는 이전에 연산된 값이 그대로 저장되어 있다. 다음으로 생각해볼 수 있는 방법은 `num_elements`라는 메서드가 호출될 때 `sequence`의 길이를 계산하는 것이다.


```python
class MyClass:
    def __init__(self, sequence):
        self.sequence = sequence

    def num_elements(self):
        return len(self.sequence)
   

foo = MyClass([1, 2, 3, 4, 5])
# sequence 속성 접근
print("num_elements 메서드 출력값: ", foo.num_elements())

# sequence 속성 할당
foo.sequence = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print("num_elements 메서드 출력값: ", foo.num_elements())
```

    num_elements 메서드 출력값:  5
    num_elements 메서드 출력값:  10
    

<br>

원하는 결과가 출력되기는 하지만 `num_elements`가 메서드라는 것이 마음에 들지 않는다. 뒤에 괄호  `()`를 써줘야만 한다. `@property` 데코레이터를 사용하면 우리가 원하는 목적을 이룰 수 있다. `@property` 데코레이터와 함께 원하는 이름으로 메서드를 만들면, 그 메서드를 속성처럼 사용할 수 있다. 


```python
class MyClass:
    def __init__(self, sequence):
        self.sequence = sequence

    @property
    def num_elements(self):
        return len(self.sequence)
   

foo = MyClass([1, 2, 3, 4, 5])
# sequence 속성 접근
print("num_elements 속성: ", foo.num_elements)

# sequence 속성 할당
foo.sequence = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print("num_elements 속성: ", foo.num_elements)
```

    num_elements 속성:  5
    num_elements 속성:  10
    

<br>
위에서 `객체.num_elements`를 호출하면 `num_elements` 메서드가 호출된다. 속성에 접근하는 것이기 때문에 뒤에 괄호를 써주지 않아도 된다. `@property`를 사용해서 속성에 접근하는 메서드를 getter 메서드라고도 부른다. 지금까지 속성에 접근할 때 원하는 연산을 하는 예시를 살펴보았다. 다음으로 속성에 값을 할당할 때, 원하는 연산을 할 수 있게 만들어보자. 예를 들어, `sequence` 속성에 리스트 자료형만 할당할 수 있도록 예외처리를 만들어보겠다.


```python
class MyClass:
    def __init__(self, sequence):
        self.sequence = sequence
        
    @property
    def sequence(self):
        return self._sequence
        
    @sequence.setter
    def sequence(self, sequence):
        if not isinstance(sequence, list):
            raise ValueError(f"sequence must be list type; got {type(sequence)}")
        self._sequence = sequence
   
foo = MyClass([1, 2, 3, 4, 5])
# sequence 속성 접근
print("sequence 속성: ", foo.sequence)

# sequence 속성 할당
foo.sequence = 100.0
print("sequence 속성: ", foo.sequence)
```

    sequence 속성:  [1, 2, 3, 4, 5]
    


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-11-e7710084d63f> in <module>
         18 
         19 # sequence 속성 할당
    ---> 20 foo.sequence = 100.0
         21 print("sequence 속성: ", foo.sequence)
    

    <ipython-input-11-e7710084d63f> in sequence(self, sequence)
         10     def sequence(self, sequence):
         11         if not isinstance(sequence, list):
    ---> 12             raise ValueError(f"sequence must be list type; got {type(sequence)}")
         13         self._sequence = sequence
         14 
    

    ValueError: sequence must be list type; got <class 'float'>


<br>

`foo.sequence = 100.0`와 같은 속성 할당이 발생할 때 `@sequence.setter`의 메서드가 호출된다. `@sequence.setter` 같은 setter를 사용하기 위해서는 먼저 `@property`를 사용하여 만들어줘야 한다. 이렇게 작성하면 좋은 점이 `__init__` 메서드에 있는 `self.sequence = sequence`를 실행할 때도 setter 메서드가 호출되면서 입력값 `sequence`가 리스트인지 아닌지 확인한다.  즉, 객체를 선언할 때 입력 받는 값의 자료형을 검증할 수 있다.


```python
foo = MyClass(100)
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-12-cce4024098a0> in <module>
    ----> 1 foo = MyClass(100)
    

    <ipython-input-11-e7710084d63f> in __init__(self, sequence)
          1 class MyClass:
          2     def __init__(self, sequence):
    ----> 3         self.sequence = sequence
          4 
          5     @property
    

    <ipython-input-11-e7710084d63f> in sequence(self, sequence)
         10     def sequence(self, sequence):
         11         if not isinstance(sequence, list):
    ---> 12             raise ValueError(f"sequence must be list type; got {type(sequence)}")
         13         self._sequence = sequence
         14 
    

    ValueError: sequence must be list type; got <class 'int'>


이렇게 `@property`와 `@property.setter`를 사용할 때 관용상 주의할 점이 있다. 먼저, 두 메서드 안에서는 최대한 관련있는 속성만 다뤄줘야 한다. 특히, `@property` 메서드 안에서는 속성들의 값을 바꾸는 행위를 자제해야 한다. 어떤 속성 A에 접근했을 뿐인데 다른 속성 B가 바뀌어버리면 사용자 입장에서는 영문도 모른채 데이터가 바뀌어 버리는 것이 된다. 그리고 두 메서드는 속성처럼 사용되기 때문에 그만큼 이해하기 쉽고 실행이 빨라야 한다. 속성에 접근/할당할 뿐인데 파일 입출력이 실행된다거나, 다른 라이브러리를 불러온다거나, 무거운 데이터베이스 처리를 한다거나 하는 것은 없어야 할 것이다.

<br>

---
