import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# file = 'H:'
# data_file = file + '/' + '7.10' + '.csv'
# csv_data = pd.read_csv(data_file)
# csv_df = pd.DataFrame(csv_data)
# 时间
# k = np.array([[0.367],[0.183],[0.161],[0.062],[0.075],[0.058],[0.061],[0.041],[0.021],[0.031],[0.013],[0.001]], dtype = np.float)
k = np.array([[0.001],[0.013],[0.031],[0.062],[0.041],[0.061],[0.058],[0.183],[0.021],[0.075],[0.367],[0.161]], dtype = np.float)
# 空间
h = np.array([[0.291, 0.176, 0.025, 0.155, 0.052, 0.071, 0.084, 0.021, 0.010, 0.032, 0.048, 0.018, 0.007, 0.03, 0.006, 0.001]], dtype = np.float)
result = np.dot(k,h)
print(result.max())

# a = np.random.uniform(0, 1, size=(12, 20))
# sns.heatmap(result, vmax=0.0155, vmin=0, cmap='GnBu')
sns.heatmap(result, vmax=0.0100, vmin=0, cmap='YlOrRd')
plt.xticks(size='small',rotation=0, fontsize=10)
plt.yticks(size='small',rotation=0, fontsize=12)
# plt.xticks([1,6,11,16,20],["lane_1","lane_6","lane_11","lane_16","lane_20"])
plt.yticks([0,2,4,6,8,10,12],["far",10,8,6,4,2,"close"])
plt.xlabel('spatial points',fontsize = 13)
plt.ylabel('time intervals', fontsize = 13)
plt.title("attention matrix of Boxue-L3", fontsize=14);
plt.savefig('attentionY3.pdf', bbox_inches='tight')
plt.show()
