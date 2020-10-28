import datetime
import time
import pandas as pd
import numpy as np
file = 'L:/data_process_time/tonghui-jianshe4-south'
max = 6

for index in range(1,32):
    date = '7.' + str(index)
    data_file = file + '/' + date + '.csv'
    csv_data = pd.read_csv(data_file)
    csv_df = pd.DataFrame(csv_data)

    a = '2017-7-' + str(index) + ' 00:00:00'
    count = np.zeros((288, max))
    for i in range(0, len(csv_df)):
        item = csv_df.iloc[i]
        str_time = item['entryTime']
        start_time = datetime.datetime.strptime(a, "%Y-%m-%d %H:%M:%S")
        end_time = start_time + datetime.timedelta(minutes=5)
        time = datetime.datetime.strptime(str_time, "%Y-%m-%d %H:%M:%S")
        for j in range(0, 288):
            if time.__ge__(start_time) and time.__lt__(end_time):
                lane = item['roadID']
                count[j][lane] = count[j][lane] + 1
                break
            start_time = end_time
            end_time = end_time + datetime.timedelta(minutes=5)

    new_df = pd.DataFrame(count[0:288, 1:], columns=['lane_1', 'lane_2', 'lane_3', 'lane_4','lane_5'])
    new_df.to_csv(date + '.csv' , encoding='utf-8')
    print(date + ' already')

for index in range(1,32):
    date = '8.' + str(index)
    data_file = file + '/' + date + '.csv'
    csv_data = pd.read_csv(data_file)
    csv_df = pd.DataFrame(csv_data)

    a = '2017-8-' + str(index) + ' 00:00:00'
    count = np.zeros((288, max))
    for i in range(0, len(csv_df)):
        item = csv_df.iloc[i]
        str_time = item['entryTime']
        start_time = datetime.datetime.strptime(a, "%Y-%m-%d %H:%M:%S")
        end_time = start_time + datetime.timedelta(minutes=5)
        time = datetime.datetime.strptime(str_time, "%Y-%m-%d %H:%M:%S")
        for j in range(0, 288):
            if time.__ge__(start_time) and time.__lt__(end_time):
                lane = item['roadID']
                count[j][lane] = count[j][lane] + 1
                break
            start_time = end_time
            end_time = end_time + datetime.timedelta(minutes=5)

    new_df = pd.DataFrame(count[0:288, 1:], columns=['lane_1', 'lane_2', 'lane_3', 'lane_4','lane_5'])
    new_df.to_csv(date + '.csv' , encoding='utf-8')
    print(date + ' already')

for index in range(1,15):
    date = '9.' + str(index)
    data_file = file + '/' + date + '.csv'
    csv_data = pd.read_csv(data_file)
    csv_df = pd.DataFrame(csv_data)

    a = '2017-9-' + str(index) + ' 00:00:00'
    count = np.zeros((288, max))
    for i in range(0, len(csv_df)):
        item = csv_df.iloc[i]
        str_time = item['entryTime']
        start_time = datetime.datetime.strptime(a, "%Y-%m-%d %H:%M:%S")
        end_time = start_time + datetime.timedelta(minutes=5)
        time = datetime.datetime.strptime(str_time, "%Y-%m-%d %H:%M:%S")
        for j in range(0, 288):
            if time.__ge__(start_time) and time.__lt__(end_time):
                lane = item['roadID']
                count[j][lane] = count[j][lane] + 1
                break
            start_time = end_time
            end_time = end_time + datetime.timedelta(minutes=5)

    new_df = pd.DataFrame(count[0:288, 1:], columns=['lane_1', 'lane_2', 'lane_3', 'lane_4','lane_5'])
    new_df.to_csv(date + '.csv' , encoding='utf-8')
    print(date + ' already')


for index in range(15,24):
    date = '9.' + str(index)
    data_file = file + '/' + date + '.csv'
    csv_data = pd.read_csv(data_file,encoding='gbk')
    csv_df = pd.DataFrame(csv_data)

    a = '2017/9/' + str(index) + ' 00:00:00'
    count = np.zeros((288, max))
    for i in range(0, len(csv_df)):
        item = csv_df.iloc[i]
        str_time = item['entryTime'] + ':00'
        start_time = datetime.datetime.strptime(a, "%Y/%m/%d %H:%M:%S")
        end_time = start_time + datetime.timedelta(minutes=5)
        time = datetime.datetime.strptime(str_time, "%Y/%m/%d %H:%M:%S")
        for j in range(0, 288):
            if time.__ge__(start_time) and time.__lt__(end_time):
                lane = item['roadID']
                count[j][lane] = count[j][lane] + 1
                break
            start_time = end_time
            end_time = end_time + datetime.timedelta(minutes=5)

    new_df = pd.DataFrame(count[0:288, 1:], columns=['lane_1', 'lane_2', 'lane_3', 'lane_4','lane_5'])
    new_df.to_csv(date + '.csv' , encoding='utf-8')
    print(date + ' already')

for index in range(24,31):
    date = '9.' + str(index)
    data_file = file + '/' + date + '.csv'
    csv_data = pd.read_csv(data_file)
    csv_df = pd.DataFrame(csv_data)

    a = '2017-9-' + str(index) + ' 00:00:00'
    count = np.zeros((288, max))
    for i in range(0, len(csv_df)):
        item = csv_df.iloc[i]
        str_time = item['entryTime']
        start_time = datetime.datetime.strptime(a, "%Y-%m-%d %H:%M:%S")
        end_time = start_time + datetime.timedelta(minutes=5)
        time = datetime.datetime.strptime(str_time, "%Y-%m-%d %H:%M:%S")
        for j in range(0, 288):
            if time.__ge__(start_time) and time.__lt__(end_time):
                lane = item['roadID']
                count[j][lane] = count[j][lane] + 1
                break
            start_time = end_time
            end_time = end_time + datetime.timedelta(minutes=5)

    new_df = pd.DataFrame(count[0:288, 1:], columns=['lane_1', 'lane_2', 'lane_3', 'lane_4','lane_5'])
    new_df.to_csv(date + '.csv' , encoding='utf-8')
    print(date + ' already')

for index in range(1,24):
    date = '10.' + str(index)
    data_file = file + '/' + date + '.csv'
    csv_data = pd.read_csv(data_file)
    csv_df = pd.DataFrame(csv_data)

    a = '2017-10-' + str(index) + ' 00:00:00'
    count = np.zeros((288, max))
    for i in range(0, len(csv_df)):
        item = csv_df.iloc[i]
        str_time = item['entryTime']
        start_time = datetime.datetime.strptime(a, "%Y-%m-%d %H:%M:%S")
        end_time = start_time + datetime.timedelta(minutes=5)
        time = datetime.datetime.strptime(str_time, "%Y-%m-%d %H:%M:%S")
        for j in range(0, 288):
            if time.__ge__(start_time) and time.__lt__(end_time):
                lane = item['roadID']
                count[j][lane] = count[j][lane] + 1
                break
            start_time = end_time
            end_time = end_time + datetime.timedelta(minutes=5)

    new_df = pd.DataFrame(count[0:288, 1:], columns=['lane_1', 'lane_2', 'lane_3', 'lane_4','lane_5'])
    new_df.to_csv(date + '.csv' , encoding='utf-8')
    print(date + ' already')
