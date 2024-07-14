class Calculator:
    def __init__(self, num1, num2):
        self.num1 = num1
        self.num2 = num2

    def add(self):
        return self.num1 + self.num2

    def subtract(self):
        return self.num1 - self.num2

    def multiply(self):
        return self.num1 * self.num2

    def divide(self):
        return self.num1 / self.num2

def validate_numbers(num1, num2):
    try:
        num1 = float(num1)
        num2 = float(num2)
        return True
    except ValueError:
        return False

def validate_operator(operator):
    if operator not in ['+', '-', '*', '/']:
        raise False
    return True

try:
    input_str = input("2개의 숫자와 연산자(+, -, *, /)를 공백으로 구분하여 입력해주세요: ")
    num1, num2, operator = input_str.split()

    if not validate_numbers(num1, num2):
        raise ValueError("num1, num2는 모두 숫자여야 합니다.")

    if not validate_operator(operator):
        raise ValueError("연산자는 다음 중 하나여야 합니다: +, -, *, /")

    calc = Calculator(float(num1), float(num2))

    if operator == '+':
        result = calc.add()
    elif operator == '-':
        result = calc.subtract()
    elif operator == '*':
        result = calc.multiply()
    elif operator == '/':
        result = calc.divide()

    print(f"The result of {num1} {operator} {num2} is: {result}")

except ValueError as e:
    print(f"Input error: {e}")

except Exception as e:
    print(f"An error occurred: {e}")