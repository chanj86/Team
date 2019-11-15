import pandas as pd
import numpy as np
import math
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.mlab import griddata
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import collections
import operator
import time
import refer_model as rm
from refer_model import AGENT
import seaborn as sns
#%%Function
# 시뮬레이션 실행
def run_experiment(N,DENS,iter=50):

	#Step0) 에이전트 생성
	agent_list = []
	perf = []
	print('.')
	for i in range(0,N):
		agent_list.append(AGENT(i))

	## Step1) 모형 초기화
	# 행위자 네트워크 생성
	rm.make_network(AGENT, agent_list, density = DENS )
	rm.find_neighborhood(AGENT, agent_list)

	# 조직 지식 업데이트
	rm.update_knw_org(AGENT, agent_list)

	# 조직 지식의 성과 업데이트 (perf_org)
	rm.update_perf_org(AGENT, agent_list, N)

	# 개별 지식의 성과 업데이트 (perf)
	rm.update_perf_ind(AGENT, agent_list, N)

	# 개별 지식 평균, 편차 업데이트
	rm.update_mu_sigma(AGENT, agent_list)

	# 개별 행위자의 효용 업데이트
	rm.update_utility(AGENT, agent_list)

	# 개별 행위자의 랭크 업데이트
	rm.update_rank(AGENT, agent_list)

	## Step2) 초기화 점검
	# 행위자 수
	if len(agent_list) == N:
		print('N={}\nDens={}\n'.format(N,DENS))
	else:
		raise ValueError('number of agent is invalid')

	## Step2)기본모형 실행
	for i in range(0,iter+1):
		rm.knowledge_iteraction(AGENT, agent_list)
		rm.knowledge_mutation(AGENT, agent_list)
		rm.update_knw_org(AGENT, agent_list)
		rm.update_perf_ind(AGENT, agent_list, N)
		rm.update_perf_org(AGENT, agent_list, N)
		rm.update_utility(AGENT, agent_list)
		rm.update_network(AGENT, agent_list)
		rm.update_mu_sigma(AGENT, agent_list)
		perf.append(AGENT.perf_org)

	print('number of agent={}\ndensity of graph={}\nperformance={}'.format(N,DENS,AGENT.perf_org))
	return (N,DENS,AGENT.perf_org,perf)

#%% 민감도 분석용 data frame 만들기
iter = 50

x = np.linspace(50,200,50) #행위자수
y = np.linspace(0.1,0.9,50) #네트워크밀도

sens_df = pd.DataFrame([],index=list(y), columns=list(x))
perf_df = []

col = 0

for i in x:
	for j in y:
		# 민감도 데이터 추가
		rlt = run_experiment(int(i),j,iter)
		sens_df[i][j] = float(rlt[2])
		# 성과 데이터 추가
		perf_df.append(rlt[3])
		col += 1

#%%
## Heat map
sens_df.to_csv('result\\sen_df.csv')

sens_df = pd.read_csv('result\\sen_df.csv')

plt.pcolor(sens_df)

plt.title('Sensitivity\nstd of performance =11.79', fontsize=10)

plt.xlabel('density', fontsize=8)

plt.ylabel('# of agent', fontsize=8)

plt.colorbar()

plt.show()

## 퍼포먼스 변화
new_perf_df = pd.DataFrame(perf_df).transpose()
new_perf_df.plot(legend=None)
plt.xlim((1,iter))
plt.ylim((0,200))
plt.xlabel('Iterations')
plt.ylabel('Organizational performance')

plt.show()

