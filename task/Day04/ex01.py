from collections import deque, namedtuple, defaultdict, Counter
import logging
import re
import threading
import multiprocessing
import time

# 숫자를 출력하는 함수
def print_numbers(lock, a):
    for i in range(5):
        with lock:
            print("i:", i)
            print("공통A: ", a.value)
            a.value += 1
        time.sleep(1)


def main():
    # 리스트 예제
    # 리스트 생성 및 기본 사용법
    fruits = ["apple", "banana", "cherry"]
    print("리스트:", fruits)

    # 리스트 요소 접근
    print("첫 번째 요소:", fruits[0])
    print("마지막 요소:", fruits[-1])

    # 리스트 요소 변경
    fruits[1] = "blueberry"
    print("요소 변경 후:", fruits)

    # 리스트 요소 추가
    fruits.append("date")
    print("요소 추가 후:", fruits)

    # 리스트 요소 삭제
    fruits.remove("apple")
    print("요소 삭제 후:", fruits)

    # 리스트 길이
    print("리스트 길이:", len(fruits))

    # 세트 예제
    # 세트 생성 및 기본 사용법
    fruits = {"apple", "banana", "cherry"}
    print("세트:", fruits)

    # 세트 요소 추가
    fruits.add("date")
    print("요소 추가 후:", fruits)

    # 세트 요소 삭제
    fruits.remove("banana")
    print("요소 삭제 후:", fruits)

    # 세트 길이
    print("세트 길이:", len(fruits))

    fruits.add("date_1")
    print("요소 추가 후:", fruits)

    # 딕셔너리 예제
    # 딕셔너리 생성 및 기본 사용법
    fruit_colors = {"apple": "red", "banana": "yellow", "cherry": "red"}
    print("딕셔너리:", fruit_colors)

    # 딕셔너리 요소 접근
    print("사과의 색:", fruit_colors["apple"])

    # 딕셔너리 요소 추가
    fruit_colors["date"] = "brown"
    print("요소 추가 후:", fruit_colors)

    # 딕셔너리 요소 삭제
    del fruit_colors["banana"]
    print("요소 삭제 후:", fruit_colors)

    # 딕셔너리 길이
    print("딕셔너리 길이:", len(fruit_colors))

    # 튜플 예제
    fruits = ("apple", "banana", "cherry")
    print("튜플:", fruits)

    # 튜플 요소 접근
    print("첫 번째 요소:", fruits[0])
    print("마지막 요소:", fruits[-1])

    # 튜플 언패킹
    fruit1, fruit2, fruit3 = fruits
    print("언패킹된 요소:", fruit1, fruit2, fruit3)

    # collections 모듈 예제
    # deque
    dq = deque(["apple", "banana", "cherry"])
    dq.append("date")
    dq.popleft()
    print("Deque:", dq)

    # namedtuple
    Fruit = namedtuple('Fruit', 'name color')
    apple = Fruit(name="apple", color="red")
    print("NamedTuple:", apple)

    # defaultdict
    dd = defaultdict(int)
    dd["apple"] += 1
    print("DefaultDict:", dd)

    # Counter
    cnt = Counter(["apple", "banana", "apple", "cherry", "banana", "banana"])
    print("Counter:", cnt)

    # 리스트 컴프리헨션 예제
    # 기본 리스트 컴프리헨션
    numbers = [1, 2, 3, 4, 5]
    squared_numbers = [x*2 for x in numbers]

    squared_numbers = []
    for x in numbers:
        squared_numbers.append(x*2)

    print("2를 곱한 숫자들:", squared_numbers)

    # 조건을 포함한 리스트 컴프리헨션
    even_numbers = [x for x in numbers if x % 2 == 0]
    print("짝수들:", even_numbers)

    # 딕셔너리 컴프리헨션 예제
    numbers = [1, 2, 3, 4, 5]

    squared_dict = {x: x*2 for x in numbers}

    print("리스트 원소에 2를 곱한 딕셔너리:", squared_dict)

    # 집합 컴프리헨션 예제
    numbers = [1, 2, 3, 4, 5]

    squared_set = {x*2 for x in numbers}

    print("리스트 원소에 2를 곱한 집합:", squared_set)

    # 중첩 리스트 컴프리헨션 예제
    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    flattened = [num for row in matrix for num in row]

    print("Flattened List:", flattened)

    # 로깅 설정
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # 로깅 객체
    logger = logging.getLogger(__name__)

    def divide(a, b):
        try:
            result = a / b
            logger.info("Division successful")
            return result
        except ZeroDivisionError:
            logger.error("Division by zero error")
            return None

    # 로깅 테스트
    print("Division successful\n", divide(10, 2))
    print("Division by zero error\n", divide(10, 0))

    logger.setLevel(logging.INFO) # 로그 레벨 설정

    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")

    # 이메일 추출
    text = "Contact us at support@example.com or sales@example.com"
    pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b"
    matches = re.findall(pattern, text)
    print("이메일:", matches)

    # HTML 태그 제거
    html = "<p>This is a <b>bold</b> paragraph.</p>"
    pattern = r"<.*?>"
    clean_text = re.sub(pattern, "", html)
    print("태그 제거 후 텍스트:", clean_text)

    # 스레드에서 사용할 락 생성
    thread_lock = threading.Lock()

    # 프로세스에서 사용할 락 생성
    process_lock = multiprocessing.Lock()

    # 전역 변수 A
    a = multiprocessing.Value('i', 0)

    # 두 개의 스레드 생성 및 시작
    thread1 = threading.Thread(target=print_numbers, args=(thread_lock, a))
    thread2 = threading.Thread(target=print_numbers, args=(thread_lock, a))
    thread1.start()
    thread2.start()
    thread1.join()
    thread2.join()

    # 전역 변수 A 초기화
    a.value = 0

    # 두 개의 프로세스 생성 및 시작
    process1 = multiprocessing.Process(target=print_numbers, args=(process_lock, a))
    process2 = multiprocessing.Process(target=print_numbers, args=(process_lock, a))
    process1.start()
    process2.start()
    process1.join()
    process2.join()

if __name__ == "__main__":
    main()