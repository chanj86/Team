import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#%%

name = ['cohe','insul','homo_high','homo_low','norm','lead']

result_perf = []
result_var = []

for n in name:
	df1 = pd.read_csv('result_{}.csv'.format(n), index_col='Unnamed: 0')
	df2 = pd.read_csv('var_{}.csv'.format(n),header=None)[1]
	result_perf.append(df1)
	result_var.append(df2)
#%%
var = pd.read_csv('var.csv', header=None)
perf = pd.read_csv('perf.csv')


#%%
plt.title('Average performance')
plt.plot(range(0,len(perf['0'])), 1/(perf['0']+1), 'black', linewidth = 3.5, label = 'Basic model with collaboration')
plt.plot(range(0,len(result_perf[0])),1/(result_perf[0]+1), '-b', label = 'High cohesiveness' )
plt.plot(range(0,len(result_perf[1])),1/(result_perf[1]+1), 'r--', label = 'High insulation' )
plt.plot(range(0,len(result_perf[2])),1/(result_perf[2]+1), 'g', linewidth = 4, label = 'High homogeneity' )
plt.plot(range(0,len(result_perf[3])),1/(result_perf[3]+1), 'orange', linewidth = 4, label = 'Low homogeneity' )
plt.plot(range(0,len(result_perf[4])),1/(result_perf[4]+1), 'cd', linewidth = 1, label = 'Lack of norms' )
plt.plot(range(0,len(result_perf[5])),1/(result_perf[5]+1), 'k:', linewidth = 2.5, label = 'Directive leadership' )
plt.xlim((1,100))
plt.xlabel('Time')
plt.ylim((0.4,1.3))
plt.ylabel('Performance')
plt.legend(loc='upper right')
fig = plt.gcf()
fig.set_size_inches((8,7.3))
plt.show()
#%%
plt.title('Average variance')
plt.plot(range(0,len(result_var[0])),result_var[0], '-b', label = 'High cohesiveness' )
plt.plot(range(0,len(result_var[1])),result_var[1], 'r--', label = 'High insulation' )
plt.plot(range(0,len(result_var[2])),result_var[2], 'g', linewidth = 4,  label = 'High homogeneity' )
plt.plot(range(0,len(result_var[3])),result_var[3], 'orange', linewidth = 4, label = 'Low homogeneity' )
plt.plot(range(0,len(result_var[4])),result_var[4], 'cd', linewidth = 1, label = 'Lack of norms' )
plt.plot(range(0,len(result_var[5])),result_var[5], 'k:', linewidth = 2.5, label = 'Directive leadership' )
plt.plot(range(0,len(result_var[0])), var[1], 'black', linewidth = 3.5, label = 'Basic model with collaboration')
plt.xlim((1,100))
plt.xlabel('Time')
plt.ylabel('Variance')
plt.legend(loc='upper right')
fig = plt.gcf()
fig.set_size_inches((8,4.5))
plt.show()
#%%
perf
#%%
plt.title('Varaince comparison')
plt.plot(range(0,len(result_var[0])),result_var[0], '-b', label = 'High cohesiveness' )
plt.plot(range(0,len(result_var[0])), var[1], '-r',label = 'Basic model with collaboration')
plt.legend(loc='upper right')
fig = plt.gcf()
fig.set_size_inches((8,4))
plt.show()