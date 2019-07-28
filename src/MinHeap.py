class MinHeap:
    def __init__(self,  k):
        self.count = 0
        self.array = [None] * (k+1)

    def swap(self, i, j):
        self.array[i], self.array[j] = self.array[j], self.array[i]

    def rise(self, k):
        while k > 1 and self.array[k][1] < self.array[k//2][1]:
            self.swap(k, k//2)
            k //= 2

    def push(self, item):
        self.count += 1
        self.array[self.count] = item
        self.rise(self.count)

    def smallest_child(self, k):
        if 2*k == self.count or self.array[2*k][1] < self.array[2*k+1][1]:
            return 2*k
        else:
            return 2*k+1

    def sink(self, k):
        while 2*k <= self.count:
            child = self.smallest_child(k)
            if self.array[k][1] <= self.array[child][1]:
                break
            self.swap(child, k)
            k = child

    def pop(self):
        item = [self.array[1][0], self.array[1][1]]
        self.swap(1, self.count)
        self.count -= 1
        self.sink(1)
        return item

    def __str__(self):
        res = ""
        for i in range(1, self.count):
            res += '(' + self.array[i][0] + ', ' + str(self.array[i][1]) + '), '
        res += '(' + self.array[self.count][0] + ', ' + str(self.array[self.count][1]) + ')'
        return res

    def __len__(self):
        return self.count

def heapSort(aList):
    aHeap = MinHeap(len(aList))
    outputList = [None] * len(aList)
    for i in range(len(aList)):
        aHeap.push(aList[i])

    for i in range(len(aHeap)):
        outputList[i] = aHeap.pop()

    return outputList

if __name__ == "__main__":
    # firstheap = MinHeap()
    # firstheap.push('abcd', 5)
    # firstheap.push('cdef', 2)
    # firstheap.push('lmop', 1)
    # firstheap.push('dsa', 6)
    # print(firstheap)
    #
    # firstheap.pop()
    # print(firstheap)
    # firstheap.pop()
    # print(firstheap)

    print(heapSort([['abcd', 5], ['cdef', 2], ['lmop', 1], ['dsa', 6]]))