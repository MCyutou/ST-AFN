# 1
# July 1 ~ July 15
# encoding:utf-8
import pandas as pd
road_a = '建设一路南'


for index in range(1,32):
    time = '7.' + str(index)
    data_file = 'L:/tonghui/6. tonghui-jianshe1/' + time + '.csv'
    # data_file = '/media/B311/My Passport/video_struct/result_video_oss_v1_vehicle_' + time + '.csv'
    # data_file = 'I:/video_struct/result_video_oss_v1_vehicle_' + time + '.csv'
    csv_data = pd.read_csv(data_file)
    csv_df = pd.DataFrame(csv_data)
    columnAll = []
    for i in range(0, len(csv_df)):
        item = csv_df.iloc[i]
        position = item['cameraPosition']
        if road_a in position:
            column = []
            column.append(item['vehicleID'])
            column.append(item['entryTime'])
            column.append(item['leaveTime'])
            column.append(item['vehicleType'])
            column.append(item['cameraID'])
            column.append(item['cameraPosition'])
            column.append(item['directionID'])
            column.append(item['roadID'])
            column.append(item['turnID'])
            column.append(item['memo'])
            print(column)
            columnAll.append(column)
    new_df = pd.DataFrame(columnAll,
                          columns=['vehicleID', 'entryTime', 'leaveTime', 'vehicleType', 'cameraID', 'cameraPosition',
                                   'directionID', 'roadID', 'turnID', 'memo'])
    new_df.to_csv(time + '.csv', encoding='utf-8')

for index in range(1,32):
    time = '8.' + str(index)
    data_file = 'L:/tonghui/6. tonghui-jianshe1/' + time + '.csv'
    # data_file = '/media/B311/My Passport/video_struct/result_video_oss_v1_vehicle_' + time + '.csv'
    # data_file = 'I:/video_struct/result_video_oss_v1_vehicle_' + time + '.csv'
    csv_data = pd.read_csv(data_file)
    csv_df = pd.DataFrame(csv_data)
    columnAll = []
    for i in range(0, len(csv_df)):
        item = csv_df.iloc[i]
        position = item['cameraPosition']
        if road_a in position:
            column = []
            column.append(item['vehicleID'])
            column.append(item['entryTime'])
            column.append(item['leaveTime'])
            column.append(item['vehicleType'])
            column.append(item['cameraID'])
            column.append(item['cameraPosition'])
            column.append(item['directionID'])
            column.append(item['roadID'])
            column.append(item['turnID'])
            column.append(item['memo'])
            print(column)
            columnAll.append(column)
    new_df = pd.DataFrame(columnAll,
                          columns=['vehicleID', 'entryTime', 'leaveTime', 'vehicleType', 'cameraID', 'cameraPosition',
                                   'directionID', 'roadID', 'turnID', 'memo'])
    new_df.to_csv(time + '.csv', encoding='utf-8')

for index in range(1,31):
    time = '9.' + str(index)
    data_file = 'L:/tonghui/6. tonghui-jianshe1/' + time + '.csv'
    # data_file = '/media/B311/My Passport/video_struct/result_video_oss_v1_vehicle_' + time + '.csv'
    # data_file = 'I:/video_struct/result_video_oss_v1_vehicle_' + time + '.csv'
    csv_data = pd.read_csv(data_file)
    csv_df = pd.DataFrame(csv_data)
    columnAll = []
    for i in range(0, len(csv_df)):
        item = csv_df.iloc[i]
        position = item['cameraPosition']
        if road_a in position:
            column = []
            column.append(item['vehicleID'])
            column.append(item['entryTime'])
            column.append(item['leaveTime'])
            column.append(item['vehicleType'])
            column.append(item['cameraID'])
            column.append(item['cameraPosition'])
            column.append(item['directionID'])
            column.append(item['roadID'])
            column.append(item['turnID'])
            column.append(item['memo'])
            print(column)
            columnAll.append(column)
    new_df = pd.DataFrame(columnAll,
                          columns=['vehicleID', 'entryTime', 'leaveTime', 'vehicleType', 'cameraID', 'cameraPosition',
                                   'directionID', 'roadID', 'turnID', 'memo'])
    new_df.to_csv(time + '.csv', encoding='utf-8')

for index in range(1,24):
    time = '10.' + str(index)
    data_file = 'L:/tonghui/6. tonghui-jianshe1/' + time + '.csv'
    # data_file = '/media/B311/My Passport/video_struct/result_video_oss_v1_vehicle_' + time + '.csv'
    # data_file = 'I:/video_struct/result_video_oss_v1_vehicle_' + time + '.csv'
    csv_data = pd.read_csv(data_file)
    csv_df = pd.DataFrame(csv_data)
    columnAll = []
    for i in range(0, len(csv_df)):
        item = csv_df.iloc[i]
        position = item['cameraPosition']
        if road_a in position:
            column = []
            column.append(item['vehicleID'])
            column.append(item['entryTime'])
            column.append(item['leaveTime'])
            column.append(item['vehicleType'])
            column.append(item['cameraID'])
            column.append(item['cameraPosition'])
            column.append(item['directionID'])
            column.append(item['roadID'])
            column.append(item['turnID'])
            column.append(item['memo'])
            print(column)
            columnAll.append(column)
    new_df = pd.DataFrame(columnAll,
                          columns=['vehicleID', 'entryTime', 'leaveTime', 'vehicleType', 'cameraID', 'cameraPosition',
                                   'directionID', 'roadID', 'turnID', 'memo'])
    new_df.to_csv(time + '.csv', encoding='utf-8')
