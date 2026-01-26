import gevent 
import requests
from bs4 import BeautifulSoup
from gevent import monkey
monkey.patch_all()


def taskA():
    for i in range(5):
        print("Task A - Step", i)
        gevent.sleep(1)

def taskB():
    for i in range(5):
        print("Task B - Step", i)
        gevent.sleep(1)
        if i == 2:
           response =requests.get("https://www.google.com")
           response.encoding = 'utf-8'
           print("Fetched page length:", len(response.text))


def taskC():
    for i in range(5):
        print("Task C - Step", i)
        gevent.sleep(1)

if __name__ == '__main__':
    print("Starting tasks...")
    task1 = gevent.spawn(taskA)
    task2 = gevent.spawn(taskB)
    task3 = gevent.spawn(taskC)
    task1.join()
    task2.join()
    task3.join()