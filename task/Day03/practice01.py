import re

def validate_age(age):
    try:
        age = int(age)
        if age > 0:
            return True
        else:
            return False
    except ValueError:
        return False

def validate_email(email):
    if re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
        return True
    else:
        return False

def main():
    try:
        name = input("이름을 입력해주세요: ")

        age = input("나이를 입력해주세요: ")
        if not validate_age(age):
            raise ValueError("나이는 양수인 숫자여야 합니다.")

        email = input("이메일을 입력해주세요: ")
        if not validate_email(email):
            raise ValueError("올바르지 않은 이메일 형식입니다.")

        user_data = {
            'name': name,
            'age': age,
            'email': email
        }
        print(user_data)
    except ValueError as e:
        print(f"입력 오류: {e}. 다시 시도해주세요.")
    except Exception as e:
        print(f"알 수 없는 오류가 발생했습니다: {e}. 다시 시도해주세요.")
        
