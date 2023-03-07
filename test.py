import random
from genetic_algorithm import easyDecode,check_feasibility2
p1 = ['X1', 16, 2, 9, 5, 14,
      'X2', 0, 15, 13, 4, 8, 11, 3,
      'X3', 12, 17, 7, 18, 23, 6, 10, 22, 1, 20, 19,
      'X4', 21, 24]

p2 = ['X1', 15, 2,
      'X2', 3, 18, 16, 13, 8, 11, 4, 10, 1,
      'X3', 12, 6, 23, 22, 14, 20, 9, 0, 7, 17, 5,
      'X4', 19, 24, 21]



'''
用分段交配
1班 x 1班
2、3班 x 2、3班
4班 x 4班

1. 迴圈遍歷所有工班
2. 找出p1 p2中較短的分段，從中產生一個隨機切點
'''
def crossover_new(p1,p2):
    child = []

    # 找出每個'X'的位置
    p1_idx1 = p1.index('X2')
    p1_idx2 = p1.index('X4')

    p2_idx1 = p2.index('X2')
    p2_idx2 = p2.index('X4')

    # 切片取出每個分段
    p1_seg1 = p1[:p1_idx1]
    p1_seg2 = p1[p1_idx1:p1_idx2]
    p1_seg3 = p1[p1_idx2:]

    p2_seg1 = p2[:p2_idx1]
    p2_seg2 = p2[p2_idx1:p2_idx2]
    p2_seg3 = p2[p2_idx2:]

    # 分段交配
    def seg_cross(s1,s2):

        child_seg = []

        # 找到比較短的長度
        seg_len = len(s1) if len(s1) < len(s2) else len(s2)

        # 產生交配點
        cut_point = random.randint(0,seg_len-1)

        # 進行交配
        child_seg = s1[:cut_point]

        for gene in s2:
            if gene not in child_seg:
                child_seg.append(gene)

        return child_seg

    # 刪除重複數字
    def keepOne(original_list):
        new_list = []
        appeared_set = set()

        for num in original_list:

            # 數字尚未出現過
            if num not in appeared_set:
                new_list.append(num)
                appeared_set.add(num)

            # 數字已經存在
            else:
                if random.random() < 0.5:

                    # 把原本的數字刪掉再加入，讓他跑到最後面的位置
                    new_list.remove(num)
                    new_list.append(num)

        return new_list

    child_seg1 = seg_cross(p1_seg1,p2_seg1)
    child_seg2 = seg_cross(p1_seg2,p2_seg2)
    child_seg3 = seg_cross(p1_seg3,p2_seg3)

    child = child_seg1 + child_seg2 + child_seg3
    child = keepOne(child)

    return child




# p3,p4 = easyDecode(p3),easyDecode(p4)
