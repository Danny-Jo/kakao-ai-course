import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 로깅 예제
logging.info("정보 메시지")
logging.warning("경고 메시지")
logging.error("에러 메시지")

# 함수 예제
def divide(a, b):
    try:
        return a / b
    except ZeroDivisionError as e:
        logging.error("0으로 나눌 수 없습니다.", exc_info=True)

# 로깅 예제
logging.info("divide 함수 시작")
result = divide(10, 0)
logging.info("divide 함수 끝")
