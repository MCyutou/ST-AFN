road_a = '市心'
road_b = '彩虹'
for index in range(1,16):
    time = '7.' + str(index)
    data_file = '/media/B311/My Passport/video_struct/result_video_oss_v1_vehicle_' + time + '.csv'
    # data_file = 'I:/video_struct/result_video_oss_v1_vehicle_' + time + '.csv'
    csv_data = pd.read_csv(data_file)
    csv_df = pd.DataFrame(csv_data)
    columnAll = []
    for i in range(0, len(csv_df)):
        item = csv_df.iloc[i]
        position = item['cameraPosition']
        if road_a in position and road_b in position:
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
    new_df.to_csv(road_a+'-'+road_b+time+'.csv', encoding='utf-8')


# 跳过9月17日
road_a = '市心'
road_b = '彩虹'
for index in range(16,31):
    if index == 17:
        continue
    time = '9.' + str(index)
    data_file = '/media/B311/My Passport/video_struct/result_video_oss_v1_vehicle_' + time + '.csv'
    # data_file = 'I:/video_struct/result_video_oss_v1_vehicle_' + time + '.csv'
    csv_data = pd.read_csv(data_file)
    csv_df = pd.DataFrame(csv_data)
    columnAll = []
    for i in range(0, len(csv_df)):
        item = csv_df.iloc[i]
        position = item['cameraPosition']
        if road_a in position and road_b in position:
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
    new_df.to_csv(road_a+'-'+road_b+time+'.csv', encoding='utf-8')


# 10月23日处理
road_a = '市心'
road_b = '彩虹'
for index in range(13,25):
    time = '10.' + str(index)
    data_file = '/media/B311/My Passport/video_struct/result_video_oss_v1_vehicle_' + time + '.csv'
    # data_file = 'I:/video_struct/result_video_oss_v1_vehicle_' + time + '.csv'
    csv_data = pd.read_csv(data_file)
    csv_df = pd.DataFrame(csv_data)
    columnAll = []
    if index == 13:
        for i in range(0,len(csv_df)):
            item = csv_df.iloc[i]
            position = item[5]
            if road_a in position and road_b in position:
                column = []
                column.append(item[0])
                column.append(item[1])
                column.append(item[2])
                column.append(item[3])
                column.append(item[4])
                column.append(item[5])
                column.append(item[6])
                column.append(item[7])
                column.append(item[8])
                column.append(item[9])
                print(column)
                columnAll.append(column)
        new_df = pd.DataFrame(columnAll,
                                      columns=['vehicleID', 'entryTime', 'leaveTime', 'vehicleType', 'cameraID',
                                               'cameraPosition',
                                               'directionID', 'roadID', 'turnID', 'memo'])
        new_df.to_csv(road_a + '-' + road_b + time + '.csv', encoding='utf-8')
        continue
    for i in range(0, len(csv_df)):
        item = csv_df.iloc[i]
        position = item['cameraPosition']
        if road_a in position and road_b in position:
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
    new_df.to_csv(road_a+'-'+road_b+time+'.csv', encoding='utf-8')
