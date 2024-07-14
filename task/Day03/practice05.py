import sqlite3
from practice01 import validate_age, validate_email

def create_table():
    connection = sqlite3.connect('user_data.db')
    cursor = connection.cursor()
    cursor.execute(''' 
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            age INTEGER NOT NULL,
            email TEXT NOT NULL
        )
    ''')
    connection.commit()
    connection.close()

def insert_user(name, age, email):
    connection = sqlite3.connect('user_data.db')
    cursor = connection.cursor()
    cursor.execute(''' 
        INSERT INTO users (name, age, email)
        VALUES (?, ?, ?)
    ''', (name, age, email))
    connection.commit()
    connection.close()

def get_all_users():
    connection = sqlite3.connect('user_data.db')
    connection = sqlite3.connect('user_data.db')
    cursor = connection.cursor()
    cursor.execute('SELECT * FROM users')
    users = cursor.fetchall()
    return users


def main():
    try:
        create_table()
        name = input("이름을 입력해주세요: ")

        age = input("나이를 입력해주세요: ")
        if not validate_age(age):
            raise ValueError("나이는 양수인 숫자여야 합니다.")

        email = input("이메일을 입력해주세요: ")
        if not validate_email(email):
            raise ValueError("올바르지 않은 이메일 형식입니다.")

        insert_user(name, age, email)
        print(f"데이터베이스에 {name}이/가 정상적으로 저장되었습니다.")

        users = get_all_users()
        print("데이터베이스에 저장된 유저들: ")
        for user in users:
            print(user)

    except ValueError as e:
        print(f"입력 오류: {e}. 다시 시도해주세요.")

    except Exception as e:
        print(f"알 수 없는 오류가 발생했습니다: {e}. 다시 시도해주세요.")

if __name__ == "__main__":
    main()