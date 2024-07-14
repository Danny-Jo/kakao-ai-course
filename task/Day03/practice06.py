def main():
    while True:
        try:
            num = input('숫자를 입력하세요: ')
            num = float(num)
            result = 10 / num
            print(f'결과: {result}')
            break
        except ValueError:
            print("유효한 숫자를 입력하세요.")
        except ZeroDivisionError:
            print("0으로 나눌 수 없습니다. 다른 숫자를 입력하세요.")
    
if __name__ == "__main__":
    main()
        