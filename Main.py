import os
from AlgoSift import CreateSiftImgByList
from AlgoKanade import KanadeByList
from AlgoGrimson import GrimsonByList

if __name__ == "__main__":
    listSorted = os.listdir('Data')
    listSorted = sorted(listSorted, key=lambda x: (lambda a, _, b: (a, int(b)) if b.isdigit() else (float("inf"), x))(*x.partition(" ")))
#    listSorted = sorted(listDataName)
    print(listSorted)
    try:
        os.mkdir('DataSift')
    except OSError:
        pass
    CreateSiftImgByList(listSorted)

    try:
        os.mkdir('DataKanade')
    except OSError:
        pass
    KanadeByList(listSorted)

    try:
        os.mkdir('DataGrimson')
    except OSError:
        pass
    GrimsonByList(listSorted)
