from threading import Thread
from time import sleep

n=0

def task(name):
    print(f'Thread {name}: starting')
    global n
    for i in range(10000000):
        n += 1   
    sleep(1)
    print(f'Thread {name}: finishing')


def task2(name):
    print(f'Thread {name}: starting')
    global n
    for i in range(10000000):
        n -= 1   
    sleep(1)
    print(f'Thread {name}: finishing')

if __name__ == '__main__':
    thread1 = Thread(target=task, name='thread1', args=('task1',))
    thread2 = Thread(target=task2, name='thread2', args=('task2',))
    thread1.start()
    thread2.start()
    thread1.join()
    thread2.join()
    print('All threads finished and n = ', n) 

list1=[0]*10
print(list1)