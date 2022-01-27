# Implement Queue using List(Functions)
#c++ exaample https://juanchopanzacpp.wordpress.com/2013/02/26/concurrent-queue-c11/
class ConcurrentQueue:
    def __init__(self, size):
            self.size = size
            self.Queue = []
    
    def Enqueue(self, data):
        if len(self.Queue)==self.size: # check wether the stack is full or not
            print("Queue is Full!!!!")
        else:
            self.Queue.append(data)
            print(data,"is added to the Queue!")
    
    def dequeue(self):
        if not self.Queue:# or if len(stack)==0
            print("Queue is Empty!!!")
        else:
            e=self.Queue.pop(0)
            print("element removed!!:",e)
            return(e)
    
    def display(self):
        print(self.Queue)

if __name__ == '__main__':
    
    Q = ConcurrentQueue(20)
    
    Q.Enqueue(1)
    Q.display()
    Q.Enqueue(2)
    Q.display()
    p = Q.dequeue()
    Q.display()
