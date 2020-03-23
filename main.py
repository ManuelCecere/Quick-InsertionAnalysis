import numpy as np
from timeit import default_timer as timer
import matplotlib.pyplot as plt


def insertionSort(A):
    for j in range(1, len(A)):
        key = A[j]
        i = j - 1
        while i >= 0 and A[i] > key:
            A[i + 1] = A[i]
            i = i - 1
        A[i + 1] = key


def quickSort(A, p, r):
    if p < r:
        q = partition(A, p, r)
        quickSort(A, p, q - 1)
        quickSort(A, q + 1, r)


def partition(A, p, r):
    x = A[r]
    i = p - 1
    for j in range(p, r):
        if A[j] <= x:
            i = i + 1
            A[i], A[j] = A[j], A[i]
    A[i + 1], A[r] = A[r], A[i + 1]
    return i + 1


def generateOrdVector(N):
    A = np.array((range(N)))
    return A


def generateBackVector(N):
    A = np.array(range(N - 1, -1, -1))
    return A


def generateRandVector(N):
    A = np.random.randint(0, N, N)
    return A


def generateQsBestCase(N):
    A = generateOrdVector(N)
    BestCaseAux(A, 0, N - 1)
    return A


def BestCaseAux(A, p, r):
    if r < p:
        return
    q = int((p + r) / 2)

    BestCaseAux(A, p, q - 1)
    BestCaseAux(A, q + 1, r)
    A[q], A[r] = A[r], A[q]


def testInsertionSort():
    totAvgOrd = []
    totAvgRand = []
    totAvgBack = []
    size = range(60, 960, 60)
    k = 40

    for s in size:
        avgOrd = 0
        avgRand = 0
        avgBack = 0
        for i in range(0, k):
            A = generateRandVector(s)
            start = timer()
            insertionSort(A)
            avgRand += timer() - start

            B = generateOrdVector(s)
            start = timer()
            insertionSort(B)
            avgOrd += timer() - start

            C = generateBackVector(s)
            start = timer()
            insertionSort(C)
            avgBack += timer() - start
        avgRand = avgRand / k
        avgOrd = avgOrd / k
        avgBack = avgBack / k

        totAvgRand.append(avgRand * 1000)
        totAvgOrd.append(avgOrd * 1000)
        totAvgBack.append(avgBack * 1000)

    plt.plot(size, totAvgRand, label="Random Vector")
    plt.plot(size, totAvgBack, label="Backwards Vector")
    plt.plot(size, totAvgOrd, label="Ordained Vector")
    plt.legend()
    plt.ylabel("Time in ms")
    plt.xlabel("Input's dimension ")
    plt.show()


def testQuickSort():
    totAvgOrd = []
    totAvgRand = []
    totAvgBack = []
    totAvgBest = []
    size = range(60, 960, 60)
    k = 40

    for s in size:
        avgOrd = 0
        avgRand = 0
        avgBack = 0
        avgBest = 0
        for i in range(0, k):
            A = generateRandVector(s)
            start = timer()
            quickSort(A, 0, s - 1)
            avgRand += timer() - start

            B = generateOrdVector(s)
            start = timer()
            quickSort(B, 0, s - 1)
            avgOrd += timer() - start

            C = generateBackVector(s)
            start = timer()
            quickSort(C, 0, s - 1)
            avgBack += timer() - start

            D = generateQsBestCase(s)
            start = timer()
            quickSort(D, 0, s - 1)
            avgBest += timer() - start
        avgRand = avgRand / k
        avgOrd = avgOrd / k
        avgBack = avgBack / k
        avgBest = avgBest / k

        totAvgRand.append(avgRand * 1000)
        totAvgOrd.append(avgOrd * 1000)
        totAvgBack.append(avgBack * 1000)
        totAvgBest.append(avgBest * 1000)
    plt.plot(size, totAvgRand, label="Random Vector")
    plt.plot(size, totAvgBack, label="Backwards Vector")
    plt.plot(size, totAvgOrd, label="Ordained Vector")
    plt.plot(size, totAvgBest, label="Best Case Vector")
    plt.legend()
    plt.ylabel("Time in ms")
    plt.xlabel("Input's dimension ")
    plt.show()

def main():
    testQuickSort()
    testInsertionSort()


if __name__ == '__main__':
    main()