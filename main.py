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

    plt.plot(size, totAvgRand, label="Valori casuali")
    plt.plot(size, totAvgBack, label="Valori ordinati al contrario")
    plt.plot(size, totAvgOrd, label="Valori ordinati")
    plt.legend()
    plt.ylabel("Tempo in ms")
    plt.xlabel("Dimensione dell'input")
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

    plt.plot(size, totAvgRand, label="Valori casuali")
    plt.plot(size, totAvgBack, label="Valori ordinati al contrario")
    plt.plot(size, totAvgOrd, label="Valori ordinati")
    plt.plot(size, totAvgBest, label="Valori nel caso migliore")
    plt.legend()
    plt.ylabel("Tempo in ms")
    plt.xlabel("Dimensione dell'input")
    plt.show()


def insertionVsQuick():
    totAvgOrdI = []
    totAvgRandI = []
    totAvgOrdQ = []
    totAvgRandQ = []
    size = range(60, 460, 60)
    k = 40

    for s in size:
        avgOrdI = 0
        avgRandI = 0
        avgOrdQ = 0
        avgRandQ = 0
        for i in range(0, k):
            A = generateRandVector(s)
            A2 = np.array(A)
            start = timer()
            insertionSort(A)
            avgRandI += timer() - start

            start = timer()
            quickSort(A2, 0, s - 1)
            avgRandQ += timer() - start

            B = generateOrdVector(s)
            start = timer()
            insertionSort(B)
            avgOrdI += timer() - start

            start = timer()
            quickSort(B, 0, s - 1)
            avgOrdQ += timer() - start

        avgRandI = avgRandI / k
        avgOrdI = avgOrdI / k
        avgRandQ = avgRandQ / k
        avgOrdQ = avgOrdQ / k

        totAvgRandI.append(avgRandI * 1000)
        totAvgOrdI.append(avgOrdI * 1000)
        totAvgRandQ.append(avgRandQ * 1000)
        totAvgOrdQ.append(avgOrdQ * 1000)

    plt.plot(size, totAvgRandI, label="Insertion valori casuali ")
    plt.plot(size, totAvgOrdI, label="Insertion valori ordinati")
    plt.plot(size, totAvgRandQ, label="QuickSort valori casuali")
    plt.plot(size, totAvgOrdQ, label="QuickSort valori ordinati")

    plt.legend()
    plt.ylabel("Tempo in ms")
    plt.xlabel("Dimensione dell'input")
    plt.show()


def main():
    testQuickSort()
    testInsertionSort()
    insertionVsQuick()


if __name__ == '__main__':
    main()
