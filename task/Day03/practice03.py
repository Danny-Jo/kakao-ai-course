def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

for index, value in enumerate(fibonacci(10)):
    print(f'피보나치 숫자 {index} : {value}')