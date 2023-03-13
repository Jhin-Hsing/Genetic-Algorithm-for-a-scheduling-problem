'''
genetic algorithm for a scheduling problem

'''
from datetime import datetime
import datetime as dt
import matplotlib.pyplot as plt
import time
import random
import pandas as pd
from myFunc import transfor_lineTable,transfor_manuTable,orderLabel,createSheet
from myFunc import generate_dailySheet,manufHours,calculating_done,splitWeek
from myFunc import move_to_bottom,move_to_top,create_crntTable
import openpyxl
from openpyxl import load_workbook
pd.options.mode.chained_assignment = None

#解碼成各自的排程表陣列
def easyDecode(individual):
    '''
    將染色體切回各自排程表陣列
    例如
    [x1,2,3,4,x2,5,6,x3,9,7]
    =>
    [
        [2,3,4]
        [5,6]
        [9,7]
    ]

    做法:
    宣告tmp，每次遇到字串就表示接下來都屬於下一個陣列，把tmp加入schedule並清空
    最後還要在append一次tmp

    '''

    schedule = []
    tmp = []


    for e in individual:
        if type(e)==str:

            # 如果是字串就把tmp append到schedule並清空
            schedule.append(tmp)
            tmp = []
        else:
            # 如果是數字就 append 到 tmp
            tmp.append(e)

    # 最後還要把tmp加入一次
    schedule.append(tmp)

    #刪除第一個空list
    schedule = schedule[1:]

    return schedule

#將二維陣列轉換為order的df
def convert_to_dataFrame(schedule,order):
    '''
    傳入代表順序的index
    回傳dataFrame的製令單

    '''
    df_schedule = []
    crew_num = 1

    for new_index in schedule:
        df = pd.DataFrame()
        df = order.reindex(new_index)
        df_schedule.append(df)
        crew_num += 1

    return df_schedule

#可行解檢查
def check_feasibility(individual,order,manuTable,lineTable,dailySheet,fillTable_path):
    '''
    (a) 該工班能不能做這張單
    (b) 做完會不會超過交期


    (a) test1 :
        1.簡單解碼後去逐個比對，只要發現一筆違反限制就break

    (b) test2:
        1.將完整製令單依照各班排程表切分成df並放入df_schedule
        2.傳入calculating_done 計算完工時間
        3.確認是否所有訂單都沒有逾期 (完工日>出貨日)，一樣只要發現一筆就break

    '''
    result = True
    test1 = True
    test2 = True
    schedule = easyDecode(individual)
    df_schedule = []

    #依序確認工班可不可以做該類型
    crew_num = 1
    for lst in schedule:
        if not test1:break

        for idx in lst:
            type = order['類型'][idx]
            if manuTable[f'製{crew_num}班11人'][type]==0:
                test1 = False
                break
        crew_num += 1

    # print(schedule)
    # print(test1)
    # quit()

    #計算各班訂單完工時間
    df_schedule = convert_to_dataFrame(schedule,order)
    crew_num = 1
    for df in df_schedule:
        df = calculating_done(str(crew_num)+'班',df,lineTable,dailySheet,fillTable_path)
        # df.to_excel(f'./debug/{crew_num}.xlsx')
        crew_num += 1


    #檢查是否有超過交期
    #前者為True才需要檢查
    if test1:
        crew_num = 1
        for df in df_schedule:
            if not test2:break
            for idx in df.index:
                complete = pd.to_datetime(df['預計完工'][idx], format='%Y/%m/%d')

                due = df['預計出貨'][idx]
                if complete >= due:
                    # print(str(crew_num)+'班',idx,complete,due)
                    test2 = False
                    # quit()
                    break
            crew_num += 1

    result = test1 and test2

    return result

#可行解檢查(final debug)
def check_feasibility2(individual,order,manuTable,lineTable,dailySheet,fillTable_path):
    '''
    (a) 該工班能不能做這張單
    (b) 做完會不會超過交期


    (a) test1 :
        1.簡單解碼後去逐個比對，只要發現一筆違反限制就break

    (b) test2:
        1.將完整製令單依照各班排程表切分成df並放入df_schedule
        2.傳入calculating_done 計算完工時間
        3.確認是否所有訂單都沒有逾期 (完工日>出貨日)，一樣只要發現一筆就break

    '''
    result = True
    test1 = True
    test2 = True
    schedule = easyDecode(individual)
    df_schedule = []

    #依序確認工班可不可以做該類型
    crew_num = 1
    for lst in schedule:
        if not test1:break

        for idx in lst:
            type = order['類型'][idx]
            if manuTable[f'製{crew_num}班11人'][type]==0:
                # print(individual)
                print(f'{idx}: 生產限制: {type},無法於{crew_num}班加工')

                # df.to_excel('./debug/vio.xlsx')
                test1 = False
        crew_num += 1

    # print(schedule)
    # print(test1)
    # quit()

    #計算各班訂單完工時間
    df_schedule = convert_to_dataFrame(schedule,order)
    crew_num = 1
    for df in df_schedule:
        df = calculating_done(str(crew_num)+'班',df,lineTable,dailySheet,fillTable_path)
        # df.to_excel(f'./debug/afet.xlsx')
        crew_num += 1


    #檢查是否有超過交期
    crew_num = 1
    for df in df_schedule:
        if not test2:break
        for idx in df.index:
            complete = pd.to_datetime(df['預計完工'][idx], format='%Y/%m/%d')

            due = df['預計出貨'][idx]
            if complete >= due:
                # print(individual)
                print(f'{crew_num}班 訂單{idx} 超過交期 預計完工:{complete} 預計出貨:{due}')
                test2 = False

                df.to_excel('./debug/vio.xlsx')
                break
        crew_num += 1

    result = test1 and test2



    return result

#初始化染色體
def init_individual(order,manuTable):
    '''
    產生染色體:
    1.讀取製令單index並打亂
    2.依照亂數分配給工班
    3.檢查工班是否可做 不可以->重新產生一個亂數
        a.因為這邊沒有計算加工時間，所以人數都用11人去看
        b.只要那張訂單分配到的工班不能做就會重複產生亂數
    4.組合成染色體的編碼格式

    '''
    individual = []
    X1 = ['X1']
    X2 = ['X2']
    X3 = ['X3']
    X4 = ['X4']
    order_list = order.index.tolist()
    random.shuffle(order_list)


    for idx in order_list:
        type = order['類型'][idx]
        while(True):
            rnd = round(random.uniform(0,1),2)
            if rnd<=0.25:
                if manuTable['製1班11人'][type]==1:
                    X1.append(idx)
                    break
                else:continue

            elif rnd<=0.5:
                if manuTable['製2班11人'][type]==1:
                    X2.append(idx)
                    break
                else:continue

            elif rnd<=0.75:
                if manuTable['製3班11人'][type]==1:
                    X3.append(idx)
                    break
                else:continue

            elif rnd<=1:
                if manuTable['製4班11人'][type]==1:
                    X4.append(idx)
                    break
                else:continue

    for lst in [X1,X2,X3,X4]:
        for element in lst:
            individual.append(element)

    # print(X1)
    # print(X2)
    # print(X3)
    # print(X4)
    # print(individual)
    # quit()

    # 依照出貨日期排序
    individual = sort_by_dueDay(individual,order)


    return individual

#評估適應度
def fitness_evaluate(individual,order,lineTable):
    '''
    傳入個體，回傳適應值
    fitness value = 1/setup_time

    1.以easyDecode把個別個體解碼成二維陣列
    2.轉換為dataframe
    3.算出各班排程表各自的setup time再相加起來倒數就是該個體的適應度

    '''

    schedule = easyDecode(individual)

    df_schedule = convert_to_dataFrame(schedule,order)

    crew_num = 1
    sum = 0
    for df in df_schedule:
        #d
        # df['t'] = 0
        # 有可能會出現工班沒有單可以做
        if df.empty:continue

        # 計算換線時間 setup time
        type = df['類型'][df.index[0]]
        for i in df.index:
            if type != df['類型'][i]:
                sum += lineTable[type][df['類型'][i]]
                #d
                # df['t'][i] = lineTable[type][df['類型'][i]]

                type = df['類型'][i]
                
        
        # df.to_excel(f'./debug/{crew_num}換線.xlsx')
        crew_num += 1
    fitness = 1/sum

    return fitness

#輪盤選擇
def wheel_selection(POPULATION_SIZE,population,order,lineTable):
    '''
    傳入族群，回傳交配池，選擇len(population)條染色體進入mating_pool

    輪盤法:
    用累計機率的方式模擬輪盤，佔比越高的染色體在輪盤上的範圍越大，越容易被選中

    1.計算各自適應度，存入fitness_list
    2.計算佔比機率 prob_list
    3.計算累積機率 cumulative_list
    4.以迴圈遍歷 cumulative_list ，每次迴圈產生亂數pick，
        如果小於當前染色體的累積機率就選入mating_pool

    '''
    mating_pool = []
    mating_pool_idx = []
    size = POPULATION_SIZE
    # size = 100000

    # 計算所有染色體的適應度
    fitness_list = [fitness_evaluate(ind,order,lineTable) for ind in population]

    # 計算染色體的佔比機率
    prob_list = [fitness/sum(fitness_list) for fitness in fitness_list]

    # 計算染色體的累積機率
    cumulative_list = []
    for i in range(len(prob_list)):
        current = 0
        for j in range(0,i+1):
            current += prob_list[j]
        cumulative_list.append(current)

    # 進行選擇
    for i in range(size):
        pick = random.uniform(0,1)
        for j in range(len(cumulative_list)):
            if pick < cumulative_list[j]:
                mating_pool_idx.append(j)
                mating_pool.append(population[j])
                break

    # # debug
    # for i in range(len(fitness_list)):
    #     fit = round(fitness_list[i],3)
    #     prob = round(prob_list[i],3)
    #     add = round(cumulative_list[i],3)
    #     sum_fit = round(sum(fitness_list),3)
    #     sum_prob = round(sum(prob_list),3)
    #     print(f'{i} fitness:{fit}\t\t佔比:{prob}\t\t累積機率:{add}')
    # print(f"總和:{sum_fit}\t\t總和:{sum_prob}")

    # for i in set(mating_pool_idx):
    #     print(f"染色體:{i}\t數量:{mating_pool.count(mating_pool[i])}")
    # quit()

    return mating_pool

#菁英選擇
def elite_selection(POPULATION_SIZE,population,order,lineTable):
    '''
    使用菁英法

    1. 選擇20條適應度最高的染色體
    2. 剩餘20條隨機從族群內抽

    '''
    mating_pool = []

    # 計算適應度
    fitness_list = [fitness_evaluate(ind,order,lineTable) for ind in population]

    # 將染色體與適應度壓成一個list
    population_with_fitness = list(zip(population,fitness_list))
    
    # 依照適應度進行排序，key 使用 tuple 中的第1個元素
    population_with_fitness = sorted(population_with_fitness,key=lambda x:x[1],reverse=True)

    # 取出前20名的染色體
    elite_individual = population_with_fitness[:20]

    # 轉換回只有染色體的 list
    elite_individual = [x[0] for x in elite_individual]

    # 隨機選擇20條染色體
    rand_individual = random.sample(population,20)

    # 由適應度最高的20條染色體加上隨機20條組成交配池
    mating_pool = elite_individual + rand_individual

    return mating_pool

#將染色體依照交貨日期排序
def sort_by_dueDay(individual,order):

    # 解碼為部分排程表DF
    df_schedule = convert_to_dataFrame(easyDecode(individual),order)

    crew_num = 1
    new_individual = []
    for df in df_schedule:

        # 直接以 time stamp 排序
        df.sort_values('預計出貨',inplace=True)

        # 新染色體加入工班代號
        new_individual.append('X'+str(crew_num))
        crew_num += 1

        # 新染色體加入排序後的編號順序
        new_individual.extend(list(df.index))

    return new_individual

#交配
def crossover(p1,p2,order,manuTable):
    '''
    使用分段交配的方式，讓同工班的編號一起交配，確保不會違反生產限制

    1. 將 parent 進行分段
    2. 同個分段進行交配
    3. 將重複的基因隨機保留其中一個位置，其餘刪除
    4. 補齊缺漏的基因(訂單編號)

    '''

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

        # 找到比較長的長度
        seg_len = len(s1) if len(s1) > len(s2) else len(s2)

        # 產生交配點
        cut_point = random.randint(0,seg_len-1)

        # 如果交配點索引超出 list 長度就完全複製 s1
        if cut_point > len(s1)-1:
            child_seg = s1
        else:
            # 如果沒有超過上限就複製前段
            child_seg = s1[:cut_point+1]

        # 如果是第二段，因為X3位於中間，可能會被吃掉，所以一律加到前半段後面
        # 確保X3一定存在於染色體中，如果X3有重複後面也能處理
        if 'X2' in s1:
            child_seg.append('X3')

        # 如果交配點索引超出 list 長度就把 s2 整段複製進來 (不能跟前面重複)
        if cut_point > len(s2)-1:
            for gene in s2:
                if gene not in child_seg:
                    child_seg.append(gene)

        # 如果沒有超過上限就複製後段進來 (不能跟前面重複)
        else:
            for gene in s2[cut_point+1:]:
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


    # 補齊缺乏數字
    def fillLack(child):
        parent = p1

        # 找出缺乏訂單
        lack_list = []
        for idx in parent:
            if idx not in child:
                lack_list.append(idx)


        for idx in lack_list:

            # 解碼child，取得完整排程表
            schedule_child = easyDecode(child)
            df_schedule = convert_to_dataFrame(schedule_child,order)


            # 找出訂單的類型
            type = order['類型'][idx]

            # 計算可做工班的總工時
            total_dict = {}
            for crew_num in [1,2,3,4]:

                # 如果工班可做，就計算總工時
                if manuTable[f'製{crew_num}班11人'][type]==1:

                    # try:
                    total = df_schedule[crew_num-1]['加工時間'].sum()
                    # except:
                    #     print('!!!!!!!!!!!!!!!!!!!!!!!!!')
                    #     print(len(df_schedule))
                    #     quit()
                    total_dict[crew_num] = total

            # print(idx,type)
            # print(total_dict)

            # 找出總工時最小的工班
            crew = min(total_dict,key=total_dict.get)

            # 將訂單插入該工班最後面
            schedule_child[crew-1].append(idx)

            # 重新編碼回染色體
            crew_num =  1
            child = []
            for s in schedule_child:
                child.append('X'+str(crew_num))
                crew_num += 1
                child.extend(s)

        return child

    child_seg1 = seg_cross(p1_seg1,p2_seg1)
    child_seg2 = seg_cross(p1_seg2,p2_seg2)
    child_seg3 = seg_cross(p1_seg3,p2_seg3)

    child = child_seg1 + child_seg2 + child_seg3
    child = keepOne(child)
    child = fillLack(child)


    return child

#突變
def mutation(individual):
    '''
    突變
    1.同個工班內各自突變
    2.產生兩個突變點，突變點位置"不能"是工班代號，即Xn
    3.交換兩個位置的基因

    '''
    # 解碼為二維陣列
    schedule = easyDecode(individual)

    # 每個工班各自產生突變點並突變
    for crew_list in schedule:

        # 如果這班只做小於1張訂單則跳過
        if len(crew_list)<=1:continue

        # 產生突變點1
        muta_p1 = random.randint(1,len(crew_list)-1)
        while(type(crew_list[muta_p1])!=int):
            muta_p1 = random.randint(1,len(crew_list)-1)

        # 產生突變點2
        muta_p2 = random.randint(1,len(crew_list)-1)
        while(type(crew_list[muta_p2])!=int):
            muta_p2 = random.randint(1,len(crew_list)-1)

        # 交換兩突變點中的基因
        crew_list[muta_p1],crew_list[muta_p2] = \
            crew_list[muta_p2],crew_list[muta_p1]

    # 重新編碼回染色體
    crew_num =  1
    new_individual = []
    for s in schedule:
        new_individual.append('X'+str(crew_num))
        crew_num += 1
        new_individual.extend(s)

    return new_individual

#換線順序調整
def df_optimization(lineTable,df,crt,next):


    #0. 拆出兩日工單
    df.drop(crt.index,inplace=True)
    df.drop(next.index,inplace=True)
    crt_tag_1 = crt[crt['tag']==1]
    crt_tag_0 = crt[crt['tag']==0]

    #1. 對今日tag_0及下一天排序
    crt_tag_0 = crt_tag_0.sort_values('類型')
    crt = pd.concat([crt_tag_1,crt_tag_0])
    next = next.sort_values('類型')


    #2. 從tag_0中尋找今明最多類型  #! 先不管tag 直接全部一起判斷 評估哪種做法比較好
    # crtType = list(dict.fromkeys(crt_tag_0['類型'].tolist()))
    crtType = list(dict.fromkeys(crt['類型'].tolist()))

    nextType = next['類型'].tolist()
    most = None
    max = 0

    for t in crtType:
        if nextType.count(t)>max:
            max = nextType.count(t)
            most = t

    #都沒有相同類型就找最小換線時間
    if most is None:
        nextType = list(dict.fromkeys(nextType))

        minTime = 999
        minTypeCrt = None
        minTypeNext = None

        for i in crtType:
            for j in nextType:
                tmp = lineTable[i][j]
                # print(i,j,tmp)
                if tmp<minTime:
                    minTime = tmp
                    minTypeCrt = i
                    minTypeNext = j

        # print(minTime,minTypeCrt,minTypeNext)



    #3. 移動最多類型或最小換線類型並標記tag為1
    if most is not None:
        for row in crt.index:
            if crt['類型'][row]==most and crt['tag'][row]==0:
                crt = move_to_bottom(crt,row)

        for row in next.index:
            if next['類型'][row]==most:
                next=move_to_top(next,row)
                next['tag'][row]=1
    else:
        for row in crt.index:
            if crt['類型'][row]==minTypeCrt and crt['tag'][row]==0:
                crt = move_to_bottom(crt,row)

        for row in next.index:
            if next['類型'][row]==minTypeNext:
                next=move_to_top(next,row)
                next['tag'][row]=1

    #3. 移動最多類型並標記tag為1
    if most!=None:
        for row in crt.index:
            if crt['類型'][row]==most and crt['tag'][row]==0:

                crt = move_to_bottom(crt,row)

        for row in next.index:
            if next['類型'][row]==most:
                next=move_to_top(next,row)
                next['tag'][row]=1



    #4. 合併回df

    #先將這兩天合併
    tmp = pd.concat([crt,next])

    #重新給予定位用的id
    tmp['id'] = range(crt['id'].min(),next['id'].max()+1,1)
    
    #連接原df後再以index排序
    df = pd.concat([tmp,df])

    #以id重新排序
    df = df.sort_values('id')

    return df

#換線優化
def setupTimeOpti(individual,order,lineTable,dailySheet,fillTable_path):

    # 將個體轉換成排程表
    dfList = convert_to_dataFrame(easyDecode(individual),order)
    
    test=1
    for i in range(0,len(dfList)):

        df = dfList[i]

    
        # 給予定位用的編號
        df['id'] = range(1, len(df)+1)

        # 計算完工時間
        df = calculating_done(str(i+1)+'班',df,lineTable,dailySheet,fillTable_path)

        allDate = list(df['預計完工'].unique())

        df['日期'] = None
        df['tag'] = 0

        # 把預計完工寫入df欄位
        for j in df.index:
            tmp = df['預計完工'][j].split(' ')[0]
            df['日期'][j] = tmp
        
        # print('===============')

        # df.to_excel('./debug/before.xlsx')
        for date in allDate:


            #擷取當日與隔日df
            crt = df[df['日期']==date]

            nextIndex = allDate.index(date)+1

            next = None

            if nextIndex == len(allDate):
                break
            else:
                next = df[df['日期']==allDate[nextIndex]]

            # print(crt['日期'][crt.index[0]],next['日期'][next.index[0]])

            df = df_optimization(lineTable,df,crt,next)
        

        # df.to_excel('./debug/mid.xlsx')
        # quit()
        # 暫時註解
        # df.drop(['日期','tag'],axis=1,inplace=True)

        #重新計算正確預計開完工、換線時間 (換線優化後排序會調整過)
        df = calculating_done(str(i+1)+'班',df,lineTable,dailySheet,fillTable_path)

        dfList[i] = df
        # dfList[i].to_excel('./debug/after.xlsx')

        test+=1

    # 編碼回染色體
    new_individual = []
    c = 1
    for df in dfList:
        tmp = list(df.index)
        new_individual.append('X'+str(c))
        new_individual.extend(tmp)
        c += 1

    # for i in easyDecode(individual):print(i)
    # print('\n')
    # for i in easyDecode(new_individual):print(i)
    # quit()
    
    return new_individual


#計算各班ST
def st_calculation(individual,order,dailySheet,lineTable):

    #解碼染色體
    dfList = convert_to_dataFrame(easyDecode(individual),order)

    # 1.寫入數值化的出貨期限 ()"總可用工時"(從第一天到出貨前一天))
    crew_num = 0
    for df in dfList:
        df['交期'] = None
        crew_num += 1
        for i in df.index:

            # start = 第一天上班
            start = dailySheet['日期'][0]

            # end = 出貨前一天: 將預計出貨轉換為datetime，扣掉一天後轉換回str
            end = df['預計出貨'][i].to_pydatetime()-dt.timedelta(days=1)
            end = end.strftime('%Y/%m/%d')

            # due = 交期(總工時)
            due = dailySheet.set_index('日期')[str(crew_num)+'班'][start:end].sum()
            df['交期'][i] = due
    
    # 2.計算ST
    for df in dfList:
        df['ST'] = None

        # 定義參數，會隨著訂單改變值
        crn_time = 0
        crn_type = '小1-1-1-1'
        setup_time = 0

        # 遍歷所有訂單計算ST並寫入
        for i in df.index:
            due = df['交期'][i]
            manu_time = df['加工時間'][i]
            type = df['類型'][i]
            
            # 查詢換線時間
            if type==crn_type:
                setup_time = 0
            else:
                setup_time = lineTable[crn_type][type]

            df['ST'][i] = due - crn_time - manu_time - setup_time

    # 3.以ST排序訂單並編碼回染色體
    new_individual = []
    crew_num = 1
    for df in dfList:
        df = df.sort_values('ST')
        new_individual.append('X'+str(crew_num))
        new_individual.extend(list(df.index))
        crew_num += 1
    
    return new_individual


#繪製圖表
def draw(lst):

    # 以 fitness 繪製
    plt.plot(lst)
    plt.xlabel('generation')
    plt.ylabel('fitness')
    plt.savefig('./output/fitness_per_generation.jpg')
    plt.clf()

    # 轉換回setup time
    lst = [1/x for x in lst]

    # 以 setup time 繪製
    plt.plot(lst)
    plt.xlabel('generation')
    plt.ylabel('setup time')
    plt.savefig('./output/setup_time_per_generation.jpg')
    plt.clf()

#輸出最佳解
def output(individual,order,lineTable,dailySheet,fillTable_path,LOST):
    df_list = convert_to_dataFrame(easyDecode(individual),order)
    c = 1
    for i in range(len(df_list)):
        df = df_list[i]
        df['換線時間'] = None
        df['選班'] = None
        
        type = df['類型'][df.index[0]]
        for j in df.index:

            # 計算預計完工
            df = calculating_done(str(c)+'班',df,lineTable,dailySheet,fillTable_path)

            # 選班寫入 dataframe
            df['選班'][j] = str(c)+'班'

            # setup time 寫入 dataframe
            if type != df['類型'][j]:

                df['換線時間'][j] = lineTable[type][df['類型'][j]]
                type = df['類型'][j]
            else:
                df['換線時間'][j] = 0

        df_list[i] = df
        c += 1

    weekTable = splitWeek(df_list)

    createSheet(fillTable_path,dailySheet,weekTable,LOST)

#主程式
def main():

    POPULATION_SIZE = 60
    MAX_GENERATION = 100
    CROSSOVER_RATE = 0.8
    MUTATION_RATE = 0.08
    LOST = [2,2,2,3]

    '''
    資料預處理
    (a)讀取excel
    (b)換線表與可生產對照表轉換為字典
    (c)製令單df標記類型
    (d)製令單df計算加工時間

    '''

    print(f'\n[{round(time.process_time(),2)}s] 資料預處理')

    #讀取表單
    print('讀取製程資訊...')
    fillTable_path = './input_data/待填工時表單-20221121-2-2-2-3.xlsx'
    typeTable_path = './input_data/福佑電機製造部工時總攬資料_V3.xlsx'
    order = pd.read_excel('./input_data/製令單_1121-1125.xlsx')
    typeTable = pd.read_excel(typeTable_path,skiprows=1)
    dailySheet = generate_dailySheet(fillTable_path,LOST)

    #轉換字典物件
    print('轉換字典物件...')
    lineTable = transfor_lineTable('./input_data/換線表測試V3.xlsx')
    manuTable = transfor_manuTable('./input_data/工時及可生產產品對應_V2.xlsx')

    #標記類型
    print('標記製令單產品類型...')
    order = orderLabel(order,typeTable)

    #計算訂單加工時間
    print('計算製令單加工時間...')
    order = manufHours(order,typeTable_path,11)



    print('訂單筆數:',len(order))    


    '''
    初始化族群

    1. 初始化個體(individual)。
    2. 檢查個體是否為可行解，如果不是，重複步驟1，直到生成可行解。
    3. 儲存可行解到population列表中。
    4. 重複步驟1~3，直到population列表中有足夠的個體(即POPULATION_SIZE)。

    '''
    print(datetime.now().strftime("%H:%M:%S"))
    print(f'\n[{round(time.process_time(),2)}s] 初始化族群')

    population = []
    individual_num = 1
    for i in range(POPULATION_SIZE):

        # 產生染色體
        individual = init_individual(order,manuTable)
        
        # ST優化
        individual = st_calculation(individual,order,dailySheet,lineTable)
        
        

        # 維持可行解
        while(not check_feasibility(individual,order,manuTable,lineTable,dailySheet,fillTable_path)):

            individual = init_individual(order,manuTable)
            individual = st_calculation(individual,order,dailySheet,lineTable)
        
        individual_num += 1

        # 加入族群
        population.append(individual)

    # 全體進行換線優化
    for individual in population:
        individual = setupTimeOpti(individual,order,lineTable,dailySheet,fillTable_path)
            

    '''
    演化

    1. 使用一個迴圈重複執行 MAX_GENERATION 次。
    2. 在每一次迴圈中，先計算當前 population 中各個體的適應值
    3. 使用 selection 函式，選擇並產生配對池。
    4. 從配對池中取出一對個體進行交配，產生子代個體 child1、child2。
    5. 將子代基因依照出貨日期排序
    6. 重複4.~5. 直到達到族群上限，產生子代族群 offspring
    7. 重複執行第 2~6 步，直到執行 MAX_GENERATION 次為止。

    '''

    # 紀錄每代最佳 fitness
    best_fitness_list = []

    print('\n',datetime.now().strftime("%H:%M:%S"))
    print(f'[{round(time.process_time(),2)}s] 開始演化世代')
    for generation in range(MAX_GENERATION):

        # 儲存族群所有個體的 fitness
        fitness_list = []

        # 計算所有個體的 fitness
        for individual in population:
            fitness = fitness_evaluate(individual,order,lineTable)
            fitness_list.append(fitness)

        # 取出此世代中最好的 fitness
        best_fitness = max(fitness_list)

        # 將最好 fitness 加入 best_fitness_list
        best_fitness_list.append(best_fitness)
        print(f'[{round(time.process_time(),2)}s] generation:{generation+1} best fitness = {round(best_fitness,5)}\tsetup time:{1/best_fitness}')

        # 使用 selection 建立配對池
        mating_pool = elite_selection(POPULATION_SIZE,population,order,lineTable)

        '''
        交配

        1.以迴圈跑到 POPULATION_SIZE，並且每次迴圈一次都從 mating_pool 取出兩個個體 (step=2)
        2.產生隨機值
            大於突變率:直接將 parent 複製進 offspring
            小於於突變率:進行交配產生兩個 child 個體

        '''

        # 建立子代族群 offspring
        offspring = []

        # 遍歷 mating_pool，一次取出一對個體
        for i in range(POPULATION_SIZE):

            # 取出父母
            parent1,parent2 = random.sample(mating_pool,2)

            # 產生隨機值，並比較交配率
            if random.random() < CROSSOVER_RATE:

                # 進行交配，產生個體 child1、child2
                child1 = crossover(parent1,parent2,order,manuTable)
                child2 = crossover(parent2,parent1,order,manuTable)

                # ST優化
                child1 = st_calculation(child1,order,dailySheet,lineTable)
                child2 = st_calculation(child2,order,dailySheet,lineTable)

                # 換線優化
                child1 = setupTimeOpti(child1,order,lineTable,dailySheet,fillTable_path)
                child2 = setupTimeOpti(child2,order,lineTable,dailySheet,fillTable_path)

                # 依照出貨日期進行排序
                # child1 = sort_by_dueDay(child1,order)
                # child2 = sort_by_dueDay(child2,order)

            else:
                # 直接複製父母
                child1,child2 = parent1,parent2
            
            if len(offspring) >= POPULATION_SIZE:
                break

            # 將子代個體加入 offspring
            offspring.extend([child1,child2])

        '''
        突變
        1.遍歷整個族群並產生隨機值，低於突變率進行突變操作

        '''
        # 遍歷子代族群
        for i in range(len(offspring)):

            # 產生隨機值並比較突變率
            if random.random() < MUTATION_RATE:

                # 取出個體
                individual = offspring[i]

                # 進行突變
                individual = mutation(individual)

                # 放回 offspring
                offspring[i] = individual

        # 取代族群
        population = offspring
        # population = random.sample(population,20) + random.sample(offspring,20)

        # 輸出圖表
        draw(best_fitness_list)


    print(f'[{round(time.process_time(),2)}s] 可行解檢查...')
    feasib_ok = []
    for i in range(len(population)):
        individual = population[i]
        if check_feasibility2(individual,order,manuTable,lineTable,dailySheet,fillTable_path):
           feasib_ok.append(individual)
    # for individual in feasib_ok:
    #     print(individual)
    print(f"族群總數: {len(population)}  可行解: {len(feasib_ok)}")


    # 輸出最佳解
    print(f'[{round(time.process_time(),2)}s] 輸出最佳解...')
    best_ind = None
    best_fit = 0
    for ind in feasib_ok:
        fitness = fitness_evaluate(ind,order,lineTable)
        if fitness>best_fit:
            best_fit = fitness
            best_ind = ind
    
    print(f"setup time:{1/best_fit}")
    output(best_ind,order,lineTable,dailySheet,fillTable_path,LOST)


    print(f'\n[{round(time.process_time(),2)}s] 結束')

if __name__ == "__main__":
    T = 1
    for i in range(1):
        print('==========================================')
        print(f'\n[{round(time.process_time(),2)}s] 演算法執行第{T}次:')
        main()
        T += 1

