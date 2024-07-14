import threading
import time
import multiprocessing

def single_thread_function():
    for i in range(5):
        print(f"Thread running: {i}")
        time.sleep(1)

# 스레드 생성
single_thread = threading.Thread(target=single_thread_function)

# 스레드 실행
single_thread.start()

# 메인 스레드가 종료되지 않도록 대기
single_thread.join()
print("Single thread finished execution.")

def thread_function(name):
    for i in range(5):
        print(f"{name} running: {i}")
        time.sleep(1)

# 스레드 생성
thread1 = threading.Thread(target=thread_function, args=("Thread 1",))
thread2 = threading.Thread(target=thread_function, args=("Thread 2",))

# 스레드 실행
thread1.start()
thread2.start()

# 메인 스레드가 종료되지 않도록 대기
thread1.join()
thread2.join()
print("Both threads finished execution.")

def process_function(name):
    for i in range(5):
        print(f"{name} running: {i}")
        time.sleep(1)

if __name__ == "__main__":
    # 프로세스 생성
    process1 = multiprocessing.Process(target=process_function, args=("Process 1",))
    process2 = multiprocessing.Process(target=process_function, args=("Process 2",))

    # 프로세스 실행
    process1.start()
    process2.start()

    # 메인 프로세스가 종료되지 않도록 대기
    process1.join()
    process2.join()
    print("Both processes finished execution.")

