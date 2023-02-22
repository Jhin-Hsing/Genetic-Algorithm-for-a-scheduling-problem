'''
genetic algorithm for a scheduling problem

2/22~2/26 產生40條可行解

'''
import time
import random
import pandas as pd
pd.options.mode.chained_assignment = None

#換線表轉換為字典
def transfor_lineTable(lineTable_path):
    table = pd.read_excel(lineTable_path,skiprows=2)
    table = table.drop(table.columns[0],axis=1).set_index(table.columns[1])
    return table.to_dict()

#可生產對照表轉換為字典
def transfor_manuTable(manuTable_path):
    table = pd.read_excel(manuTable_path)
    table = table.drop(table.columns[0],axis=1).set_index(table.columns[1])
    return table.to_dict()

#在order上標記類型
def orderLabel(order,typeTable):
    order['類型'] = None
    for i in order.index:
        for j in typeTable.index:
            if order['產品品號'][i]==typeTable['途程品號'][j]:
                order['類型'][i] = typeTable['類別'][j]
                break
    return order

#可行解檢查
def check_feasibility(individual):
    '''
    (a) 該工班能不能做這張單
    (b) 做完會不會超過交期
    '''
    result = True

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
def fitness(population):
    return

#選擇
def selection(population):
    return

#交配
def crossover(mating_pool):
    return

#突變
def mutation(population):
    return

#主程式區塊
def main():
    POPULATION_SIZE = 40
    MAX_GENERATION = 10
    MUTATION_RATE = 0.2

    population = []

    #資料預處理
    print(f'[{round(time.process_time(),2)}s] 資料預處理')
    order = pd.read_excel('./製令單_1121-1125 - 少.xlsx')
    fillTable = pd.read_excel('./待填工時表單-20221121-2-2-2-3.xlsx')
    typeTable = pd.read_excel('./福佑電機製造部工時總攬資料_V3.xlsx',skiprows=1)
    lineTable = transfor_lineTable('./換線表測試V3.xlsx')
    manuTable = transfor_manuTable('./工時及可生產產品對應_V2.xlsx')

    order = orderLabel(order,typeTable)
    print('訂單筆數:',len(order))


    #初始化族群
    print(f'[{round(time.process_time(),2)}s] 初始化族群')

    for i in range(POPULATION_SIZE):

        individual = init_individual(order,manuTable)
        while(check_feasibility(individual)!=True):
            individual = init_individual(order,manuTable)

        population.append(individual)

    for individual in population:
        print(individual)


    #演化
    for generation in range(MAX_GENERATION):

        best_fitness = fitness(population)

        mating_pool = selection(population)

        offspring = crossover(mating_pool)

        population = offspring

        population = mutation(population)


main()