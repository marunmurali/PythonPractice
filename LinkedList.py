# python program to initialize and pring linked list

# Node class
class Node:
    def __init__(self,data):
            self.data = data
            self.next = None

class LinkedList:
    def __init__(self):
        self.head = None

    def PrintList(self):
        temp = self.head
        while(self):
            print (temp.data)
            temp = temp.next

if __name__ == '__main__':
    llist = LinkedList()

    llist.head = Node(1)
    second = Node(2)
    third = Node(3)
    fourth = Node(8)

    llist.head.next = second
    second.next = third
    third.next = fourth

    llist.PrintList()
