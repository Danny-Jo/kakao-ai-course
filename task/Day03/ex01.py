# 변수 -> 데이터를 저장하는 메모리 공간에 이름을 붙인 것
x = 10  # 정수형 변수
y = 3.14  # 실수형 변수
name = "Alice"  # 문자열 변수

print(type(x))
print(type(y))
print(type(name))

# 자료형
a = 5  # int
b = 3.2  # float
c = "Hello"  # str
d = True  # bool

print(type(a))
print(type(b))
print(type(c))
print(type(d))

# 자료형 변환
num_str = "123"
print(type(num_str))
num = int(num_str)  # 문자열을 정수로 변환
print(type(num))
print(num_str)
print(num)

# 기본 연산
a = 10
b = 3
print(a + b)  # 덧셈
print(a > b)  # 비교 연산
print(a > 5 and b < 5)  # 논리 연산

# 조건문
age = 20
if age < 18:
    print("미성년자")
elif age == 18:
    print("갓 성인")
else:
    print("성인")

# for 문 -> 리스트, 튜플 등 시퀀스 자료형의 요소를 반복
fruits = ["apple", "banana", "cherry"]
for fruit in fruits:
    print(fruit)

# for 문과 range() 함수 -> 정해진 횟수만큼 반복
for i in range(5):
    print(i)

# while 문 -> 조건이 참인 동안 반복
# count = 0
# while True:
#     print(count)
#     count += 1

# break -> 반복문을 즉시 종료, continue -> 현재 반복을 건너뛰고 다음 반복으로 이동
for i in range(10):
    if i == 5:
        break  # 5에서 반복문 종료
    print(i)
print("*"*10)
for i in range(10):
    if i % 2 == 0:
        continue  # 짝수 건너뛰기
    print(i)

# 함수
"""
def 함수이름(매개변수):
    코드 블록
    return 반환값
"""
def greet(name):
    return f"Hello, {name}!"

result = greet("Bob")
print(result)

# lambda 함수
# lambda 매개변수: 반환값
square = lambda x: x ** 2
print(square(5))

# map 함수의 인자로 사용
numbers = [1, 2, 3, 4, 5]
squares = list(map(lambda x: x ** 2, numbers))
print(squares)

# 가변 인자 (*args)
"""
def 함수이름(*args):
    코드 블록
"""
def add(*args):
    return sum(args)

print(add(1, 2, 3, 4))

# 키워드 가변 인자 (**kwargs)
"""
def 함수이름(**kwargs):
    코드 블록
"""
def introduce(**kwargs):
    for key, value in kwargs.items():
        print(f"{key}: {value}")

introduce(name="Alice", age=30, city="Seoul")

# map 함수, 모든 요소에 함수를 적용하여 새로운 리스트 반환
numbers = [1, 2, 3, 4]
squares = list(map(lambda x: x ** 2, numbers))
print(squares)

# filter 함수, 조건에 맞는 요소만 걸러내어 새로운 리스트 반환
numbers = [1, 2, 3, 4, 5]
evens = list(filter(lambda x: x % 2 == 0, numbers))
print(evens)

# reduce 함수, 모든 요소를 누적하여 단일 값을 반환
from functools import reduce
numbers = [1, 2, 3, 4]
product = reduce(lambda x, y: x * y, numbers)
print(product)

# 클래스 -> 객체를 정의하는 데 사용되는 청사진
"""
class 클래스이름:
    def __init__(self, 매개변수):
        self.속성 = 매개변수

    def 메서드이름(self, 매개변수):
        코드 블록
"""
class Dog:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def bark(self):
        return "Woof!"

# 인스턴스 생성
my_dog = Dog("Buddy", 3)
print(my_dog.name)
print(my_dog.bark())

# 상속 -> 기존 클래스(부모 클래스)를 기반으로 새로운 클래스(자식 클래스)를 정의하는 것
"""
class 부모클래스:
    # 부모 클래스의 속성과 메서드 정의

class 자식클래스(부모클래스):
    # 자식 클래스의 속성과 메서드 정의
"""
class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        raise NotImplementedError("Subclass must implement abstract method")

class Dog(Animal): # Animal 상속
    def speak(self):
        return "Woof!"

class Cat(Animal): # Animal 상속
    def speak(self):
        return "Meow!"

# 다형성 -> 동일한 인터페이스를 사용하여 서로 다른 데이터 타입의 객체를 다룰 수 있는 능력

animals = [Dog("Buddy"), Cat("Whiskers")]
for animal in animals:
    print(f"{animal.name} says {animal.speak()}")

# 매직 메서드 (Magic Methods) -> 파이썬이 내부적으로 사용하는 메서드
"""
__init__: 객체 초기화 메서드
__str__: 객체의 문자열 표현을 반환
__repr__: 객체의 공식 문자열 표현을 반환
"""
class Book:
    def __init__(self, title, author):
        self.title = title
        self.author = author

    def __str__(self):
        return f"{self.title} by {self.author}"

    def __repr__(self):
        return f"Book({self.title}, {self.author})"

my_book = Book("1984", "George Orwell")
print(str(my_book))
print(repr(my_book))

# 연산자 오버로딩 -> 파이썬의 기본 연산자를 사용자 정의 클래스에서 사용할 수 있도록 메서드를 정의
"""
__add__: + 연산자
__sub__: - 연산자
__mul__: * 연산자
__truediv__: / 연산자
"""
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)

    def __str__(self):
        return f"Vector({self.x}, {self.y})"

v1 = Vector(2, 3)
v2 = Vector(5, 7)
v3 = v1 + v2

# 이터레이터 클래스
class Counter:
    def __init__(self, max):
        # 이터레이터의 최대 값 설정
        self.max = max
        # 현재 값을 0으로 초기화
        self.current = 0

    def __iter__(self):
        # __iter__ 메서드는 이터레이터 객체 자체를 반환해야 함
        return self

    def __next__(self):
        # __next__ 메서드는 다음 값을 반환
        if self.current < self.max:
            # 현재 값을 1 증가
            self.current += 1
            # 현재 값을 반환
            return self.current
        else:
            # 현재 값이 최대 값 이상이면 StopIteration 예외 발생
            raise StopIteration

# Counter 클래스의 인스턴스를 생성
counter = Counter(5)

# 이터레이터를 사용하여 값을 반복 출력
for num in counter:
    print(num)

# 제너레이터 함수
def count_up_to(max):
    # 카운트 초기값을 1로 설정
    count = 1
    # 카운트가 최대 값보다 작거나 같은 동안 반복
    while count <= max:
        # 현재 카운트 값을 반환하고 함수의 실행 상태를 유지
        yield count
        # 카운트를 1 증가
        count += 1

# 제너레이터 객체 생성
counter = count_up_to(5)

# 제너레이터를 사용하여 값을 반복 출력
for num in counter:
    print(num)

# yield 키워드 -> 제너레이터 함수에서 값을 반환하고 함수의 실행 상태를 일시 중지
def simple_generator():
    yield 1
    yield 2
    yield 3

gen = simple_generator()
print(next(gen))
print(next(gen))
print(next(gen))

# 파일 쓰기
with open('example.txt', 'w') as file:
    file.write('Hello, world!')

# 파일 읽기
with open('example.txt', 'r') as file:
    content = file.read()
    print(content)

# 파일 쓰기
with open('example.txt', 'w') as file:
    file.write('Hello, world!')

# 여러 줄 쓰기
lines = ['First line\n', 'Second line\n', 'Third line\n']
with open('example.txt', 'w') as file:
    file.writelines(lines)

# 파일 전체 읽기
with open('example.txt', 'r') as file:
    content = file.read()
    print(content)

# 파일 한 줄씩 읽기
with open('example.txt', 'r') as file:
    line = file.readline()
    while line:
        print(line.strip())
        line = file.readline()
        print("*")

try:
    result = 10 / 0
except ZeroDivisionError as e:
    print("0으로 나눌 수 없습니다:", e)

try:
    file = open("example_.txt", "r")
    content = file.read()
except FileNotFoundError as e:
    print("파일을 찾을 수 없습니다:", e)
finally:
    file.close()

def check_positive(number):
    if number < 0:
        raise ValueError("음수는 허용되지 않습니다.")

try:
    check_positive(-5)
except ValueError as e:
    print(e)

class NegativeNumberError(Exception):
    def __init__(self, message="음수는 허용되지 않습니다."):
        self.message = message
        super().__init__(self.message)

def check_positive(number):
    if number < 0:
        raise NegativeNumberError()

try:
    check_positive(-5)
except NegativeNumberError as e:
    print(e)