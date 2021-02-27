class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

    def list2Lnode(self, l: list):
        L = None
        if len(l)>=1:
            L=ListNode(l[0],None)
        p=L
        for i in range(1,len(l)):
            s = ListNode(l[i], None)
            p.next = s
            p=s
        return L

    def print_Lnode(self, L):
        p = L
        while(p != None):
            print(p.val, end="  ")
            p = p.next
        print("end")

class Solution:
    def addTwoNumbers(self, l1, l2) -> ListNode:
        res = None
        l1_p ,l2_p = l1,l2
        carry = 0
        if l1_p != None and l2_p != None:
            data = l1_p.val+l2_p.val+carry
            carry = data//10
            data = data % 10
            res = ListNode(data, None)
            l1_p=l1_p.next
            l2_p=l2_p.next
        p = res
        while(l1_p != None and l2_p != None):
            data = l1_p.val+l2_p.val+carry
            carry = data//10
            data = data % 10
            s = ListNode(data, None)
            p.next = s
            p = s
            l1_p=l1_p.next
            l2_p=l2_p.next
        while(l1_p!=None or l2_p!=None):
            if l1_p!=None:
                data = l1_p.val+carry
                carry = data//10
                data = data % 10
                s = ListNode(data, None)
                p.next = s
                p = s
                l1_p=l1_p.next
            else:
                data = l2_p.val+carry
                carry = data//10
                data = data % 10
                s = ListNode(data, None)
                p.next = s
                p = s
                l2_p=l2_p.next
        if carry > 0:
            s = ListNode(carry, None)
            p.next = s
            p = s
        return res


class Test:
    def __init__(self):
        self.l1 = [9,9]
        self.l2 = [9,9,9,9]

    def addTwoNumbers_test(self):
        L1 = ListNode().list2Lnode(self.l1)
        L2 = ListNode().list2Lnode(self.l2)
        res = Solution().addTwoNumbers(L1, L2)
        ListNode().print_Lnode(L1)
        ListNode().print_Lnode(L2)
        ListNode().print_Lnode(res)
Test().addTwoNumbers_test()
