'''
genetic algorithm for a scheduling problem

'''
import matplotlib.pyplot as plt
import time
import random
import pandas as pd
from myFunc import transfor_lineTable,transfor_manuTable,orderLabel
from myFunc import generate_dailySheet,manufHours,calculating_done
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

            #tmp裡面有值才要append到schedule
            #如果tmp非空就會回傳True
            if tmp:
                schedule.append(tmp)
                tmp = []
        else:
            tmp.append(e)

    schedule.append(tmp)


    # print(individual)
    # for lst in schedule:
    #     for e in lst:
    #         print(e,end=' ')
    #     print()

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
    crew_num = 1
    for df in df_schedule:
        if not test2:break
        for idx in df.index:
            complete = pd.to_datetime(df['預計完工'][idx], format='%Y/%m/%d')

            due = df['預計出貨'][idx]
            if complete > due:
                # print(str(crew_num)+'班',idx,complete,due)
                test2 = False
                # quit()
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
        # 有可能會出現四班沒有單可以做
        if df.empty:continue

        # 計算換線時間 setup time
        type = df['類型'][df.index[0]]
        for i in df.index:
            if type != df['類型'][i]:
                sum += lineTable[type][df['類型'][i]]
                type = df['類型'][i]
        crew_num += 1
    fitness = 1/sum


    return fitness

#選擇
def selection(population,order,lineTable):
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
    size = len(population)
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

#選擇(菁英法)
def selection_elite(population,order,lineTable):
    mating_pool = []


    return mating_pool


#交配
def crossover(parent1,parent2):
    '''
    交配

    傳入兩個個體，回傳子代
    1.產生突變點 cut point
    2.保留 parent1 從頭到 cut point 的基因
    3.剩餘部分由 parent2 以"不跟前面重複"的方式取出基因補齊
    (*)4.以此類推產生 child2

    '''
    # 產生交配點
    cut_point = random.randint(0,len(parent1)-1)

    # 產生 child1
    child1 = parent1[:cut_point]
    for gene in parent2:
        if gene not in child1:
            child1.append(gene)

    # # 產生 child2
    # child2 = parent2[:cut_point]
    # for gene in parent1:
    #     if gene not in child2:
    #         child2.append(gene)

    return child1

#突變
def mutation(individual):
    '''
    突變
    1.產生兩個突變點，突變點位置"不能"是工班代號，即Xn
    2.交換兩個位置的基因

    '''
    muta_p1 = random.randint(1,len(individual)-1)
    while(type(individual[muta_p1])!=int):
        muta_p1 = random.randint(1,len(individual)-1)


    muta_p2 = random.randint(1,len(individual)-1)
    while(type(individual[muta_p2])!=int):
        muta_p2 = random.randint(1,len(individual)-1)

    individual[muta_p1],individual[muta_p2] = \
        individual[muta_p2],individual[muta_p1]

    return individual

#主程式
def main():

    POPULATION_SIZE = 40
    MAX_GENERATION = 100
    CROSSOVER_RATE = 0.8
    MUTATION_RATE = 0.05
    LOST = [3,3,3,3]

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
    order = pd.read_excel('./input_data/製令單_1121-1125 - 少.xlsx')
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
    print(f'\n[{round(time.process_time(),2)}s] 初始化族群')

    population = []
    individual_num = 1
    for i in range(POPULATION_SIZE):

        individual = init_individual(order,manuTable)
        while(not check_feasibility(individual,order,manuTable,lineTable,dailySheet,fillTable_path)):
            individual = init_individual(order,manuTable)

        print(f'產生第{individual_num}個可行解')

        individual_num += 1
        population.append(individual)

    # for individual in population:
    #     print(individual)
    # quit()

    '''
    演算法演化

    1. 使用一個迴圈重複執行 MAX_GENERATION 次。
    2. 在每一次迴圈中，先計算當前 population 中各個體的適應值
    3. 使用 selection 函式，選擇並產生配對池。
    4. 使用 crossover 函式，在配對池中進行交配，產生子代族群 offspring。
    5. 將 offspring 取代當前 population ，完成一次世代演化。
    6. 重複執行第 2~5 步，直到執行 MAX_GENERATION 次為止。

    '''

    # 紀錄每代最佳 fitness
    best_fitness_list = []

    # 設定計時器，避免迭代時間過久
    start_time = time.time()

    # 設定迭代時間上限
    time_limit = 1200

    print(f'\n[{round(time.process_time(),2)}s] 開始演化世代')
    for generation in range(MAX_GENERATION):

        # 確認是否超時
        if time.time() - start_time > time_limit:break

        # print('計算適應度...')
        fitness_list = []
        for individual in population:
            fitness = fitness_evaluate(individual,order,lineTable)
            fitness_list.append(fitness)

        best_fitness = max(fitness_list)

        best_fitness_list.append(best_fitness)
        print(f'generation:{generation+1} best fitness = {best_fitness}')

        #seletion
        mating_pool = selection(population,order,lineTable)

        '''
        交配

        1.以迴圈跑到 POPULATION_SIZE，並且每次迴圈一次都從 mating_pool 取出兩個個體 (step=2)
        2.產生隨機值
            大於突變率:直接將 parent 複製進 offspring
            小於於突變率:進行交配產生兩個 child 個體

        '''
        # 建立子代族群 offspring
        offspring = []
        for i in range(0,POPULATION_SIZE,2):

            # 取出父母
            parent1,parent2 = mating_pool[i],mating_pool[i+1]

            if random.random() < CROSSOVER_RATE:

                child1 = crossover(parent1,parent2)
                child2 = crossover(parent2,parent1)

                # 維持可行解
                while(not check_feasibility(child1,order,manuTable,lineTable,dailySheet,fillTable_path)):
                    child1 = crossover(parent1,parent2)

                while(not check_feasibility(child2,order,manuTable,lineTable,dailySheet,fillTable_path)):
                    child2 = crossover(parent2,parent1)

            else:
                # 直接複製父母
                child1,child2 = parent1,parent2

            # 加入 offspring
            offspring.extend([child1,child2])

        '''
        突變
        1.遍歷整個族群並產生隨機值，低於突變率進行突變操作

        '''
        for i in range(len(offspring)):
            if random.random() < MUTATION_RATE:
                individual = offspring[i]

                individual = mutation(individual)

                # 維持可行解
                # while(not check_feasibility(individual,order,manuTable,lineTable,dailySheet,fillTable_path)):
                #     individual = mutation(individual)

                # offspring[i] = individual

        # 取代族群
        population = offspring


    print('可行解檢查...')
    feasib_ok = []
    for i in range(len(population)):
        individual = population[i]
        if check_feasibility(individual,order,manuTable,lineTable,dailySheet,fillTable_path):
           feasib_ok.append(individual)
    for individual in feasib_ok:
        print(individual)
    print(f"族群總數: {POPULATION_SIZE}  可行解: {len(feasib_ok)}")

    print('繪製圖表...')
    plt.plot(best_fitness_list)
    plt.xlabel('generation')
    plt.ylabel('fitness')
    plt.savefig('./output/fitness_per_generation.jpg')


    print(f'\n[{round(time.process_time(),2)}s] 結束')

if __name__ == "__main__":
    T = 1
    for i in range(1):
        print('==========================================')
        print(f'\n[{round(time.process_time(),2)}s] 演算法執行第{T}次:')
        main()
        T += 1

