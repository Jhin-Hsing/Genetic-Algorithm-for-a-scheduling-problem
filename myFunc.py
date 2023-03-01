import pandas as pd
from openpyxl import load_workbook
from openpyxl.utils import get_column_letter
import math


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

#將當前時間換算回日期
def convertDate(dailySheet,cate,value):
    date = ''
    # print(f"\n{cate}班 當前時間: {value}")

    row = 0
    while value>0:
        minutes = dailySheet[str(cate)+'班'][row]

        # print(f"日期={dailySheet['日期'][row]}\
        #     value={value} 當日剩餘工時={minutes}")

        value -= minutes
        if value>0:
            row += 1
        else:
            value = -(value)
            break

    date = dailySheet['日期'][row]

    # print(f"轉換為日期: {date} 當日剩餘工時: {value}")
    return date

#產生每日工時表
def generate_dailySheet(fillTable_path,LOST):

    #讀取待填工時表的新增工時欄位，並加上加班時數
    #數值存放於dailySheet，一開始為二維陣列，最後轉換成dataFrame return
    #範例格式:  [
    #           [日期1,1班工時,2班工時,3班工時],
    #           [日期2,1班工時,2班工時,3班工時],
    #           [日期3,1班工時,2班工時,3班工時]
    #       ]

    dailySheet = []

    wb_fillTable = load_workbook(fillTable_path)
    ws = wb_fillTable['待填工時表']

    #最大工班數，從第六欄開始，每個製造工班佔三個欄位
    maxCate = (ws.max_column-5)//3

    #取得新增工時，並加上加班時間(乘上損耗率)
    for row in range(3,ws.max_row+1):
        date = ws['A'+str(row)].value

        #跳過空白列
        if date is None:continue

        tmp = [date]
        stdTimeList = []
        if type(ws['E'+str(row)].value) == int:
            record = 0
            ws['E'+str(row)].value = str(ws['E'+str(row)].value)
            for i in range(1,len(ws['E'+str(row)].value)+1):
                if i == len(ws['E'+str(row)].value):break
                if i%3==0:
                    stdTimeList.append(ws['E'+str(row)].value[record:i])
                    record = i
            stdTimeList.append(ws['E'+str(row)].value[record:])
        else:
            stdTimeList = ws['E'+str(row)].value.split(',')

        #查看當天各班是否有加班
        index = 8
        for cate in range(maxCate):
            overTime = ws[get_column_letter(index)+str(row)].value

            if overTime is None:
                overTime = 0
            else:
                overTime = round(overTime * 60 * (1-float(LOST[cate])/100))

            index += 3
            tmp.append(int(stdTimeList[cate])+overTime)

        dailySheet.append(tmp)

    # dailySheet轉換為dataframe

    #新增欄位
    col = ['日期']
    for cate in range(maxCate):
        col.append(str(cate+1)+'班')

    dailySheet = pd.DataFrame(dailySheet,columns=col)

    # dailySheet.to_excel('./dailySheet.xlsx')
    # quit()

    return dailySheet

#建立crnTable
def create_crntTable(dailySheet,ws_fillTable):
    crnTable = pd.DataFrame()
    crnTable = pd.DataFrame(columns=['工班','開始工作時間','目前加工類型'])
    crnTable['工班'] = dailySheet.drop('日期',axis=1).columns
    crnTable.set_index('工班',inplace=True)

    #第6欄為製造一班的剩餘工時
    x=6
    cateIdx = 1
    for i in crnTable.index:
        crnTable['開始工作時間'][i] = 0

        #讀取「前一筆訂單類型」
        crnTable['目前加工類型'][i] = ws_fillTable[get_column_letter(x)+'1'].value

        #將補正還有剩餘工時加到開始工作時間
        #如果是空白則給0
        corretHour = ws_fillTable[get_column_letter(x)+'3'].value
        if corretHour is None:
            corretHour = 0

        remaningHour = ws_fillTable[get_column_letter(x+1)+'3'].value
        if remaningHour is None:
            remaningHour = 0

        startWorkTime = corretHour + remaningHour

        crnTable['開始工作時間'][str(cateIdx)+'班'] = startWorkTime

        cateIdx += 1

        #每個製造工班有3個欄位，換下個工班時就+3
        x += 3

    return crnTable

#計算加工時間
def manufHours(order,typeTable_path,count):
    #根據人數計算加工時間，並寫入order
    #捆綁訂單的加工時間需要保留,因此先分離兩部分，等加工時間全部算完再連接
    # bundle = ['特殊類型','其餘特殊']

    # bundleOrder = order.copy()
    # for i in order.index:
    #     if order['類型'][i] in bundle:
    #         order.drop(i,inplace = True)
    #     else:
    #         bundleOrder.drop(i,inplace = True)

    # order.to_excel('input/debug/order.xlsx')
    # bundleOrder.to_excel('input/debug/bun.xlsx')

    #新增欄位 台/分，並且重置為None
    order['台/分'] = None
    order['加工時間'] = None

    #讀取機種對照表
    typeTable = load_workbook(typeTable_path)
    ws = typeTable['產品途程明細表 (主檔) 20190214']


    #根據人數選擇縱軸欄位
    #固定欄位:
    # 11 : 'E'
    # 8  : 'G'
    # 6  : 'H'

    col = ''
    if count==11:
        col = 'E'
    elif count==8:
        col = 'G'
    elif count==6:
        col = 'H'


    #迴圈寫入對應類型加工時間
    for i in order.index:
        for row in range(3,ws.max_row):
            if order['產品品號'][i] == ws['A'+str(row)].value:

                #讀取對應單位工時
                hour = ws[col + str(row)].value
                if type(hour)==str:
                    continue

                # 加工時間(分) = 產量 / (一分鐘做幾個)
                time = math.ceil(order['產量'][i] / hour)
                order['台/分'][i] = hour
                order['加工時間'][i] = time
                break

    #連接捆綁訂單
    # order = pd.concat([order,bundleOrder])


    order.reset_index(drop=True,inplace=True)


    return order

#計算開工與完工
def calculating_done(cate,df,lineTable,dailySheet,fillTable_path):

    #讀取待填工時表
    ws_fillTable = load_workbook(fillTable_path)['待填工時表']


    #初始化crnTable
    crnTable = pd.DataFrame()
    crnTable = create_crntTable(dailySheet,ws_fillTable)



    #更新排程表的換線時間欄位
    type = crnTable['目前加工類型'][cate]

    for i in df.index:
        if df['類型'][i]!=type:
            type = df['類型'][i]


    #更新預計開工與完工
    for i in df.index:

        #當前時間加上換線時間
        lineTime = 0
        type = crnTable['目前加工類型'][cate]
        if df['類型'][i] != type:
            lineTime = lineTable[df['類型'][i]][type]

        crnTable['開始工作時間'][cate] += lineTime


        #當前時間加上 "加工時間" 與更新當前類型
        ori = convertDate(dailySheet,cate[:1],crnTable['開始工作時間'][cate])

        crnTable['開始工作時間'][cate] += df['加工時間'][i]
        crnTable['目前加工類型'][cate] = df['類型'][i]

        #完工日期存入crn
        crn = convertDate(dailySheet,cate[:1],crnTable['開始工作時間'][cate])
        df['預計開工'][i] = ori
        df['預計完工'][i] = crn


    return df
