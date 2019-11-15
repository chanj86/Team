import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random
import collections
import operator
import copy
import seaborn as sns
import time
# Parameters
## Organization
# Organizational Schema
performance = []
org_know = []

# Organizational conccurence
conccurence = []

# Insulation
theta = 0

#Ratio of layers
r_top = 0.1
r_middle = 0.2
r_bottom = 0.7

#Influence between layers
h_sup = 0

#number of agent
N = 50
#number of knowledge
M= 20

#Agent List
agent_list = []

#max iteration
max_iter = 100 #조직의사결정 수행 기간
ITER = 10  #모형 타당성 검증을 위한 반복

model_type = 'on'



# Agent
## Agent Class
class Agent:

	number_of_agent = 0
	homogeneity = 0
	adj = np.ones((number_of_agent, number_of_agent))

	def __init__(self,id,learning,collaboration,creativity,M,mu,std):
		self.id = id
		self.mu = mu
		self.std = std
		self.neiborhood = 0
		self.collaboration = collaboration
		self.learning = learning
		self.creativity = creativity
		self.knowledge = [round(x) for x in np.random.randint(-50*M,50*M,M)] #지식분포의 평균은 -5~5 사이, 분산은 0~5 사이
		Agent.number_of_agent += 1
		if np.random.rand() > 0.9:
			self.hierarchy = 2
		elif np.random.rand() >0.7:
			self.hierarchy = 1
		else:
			self.hierarchy = 0


	def __repr__(self):
		return 'AGENT{}'.format(self.id)

	def basic_info(self):
		return 'id:{}, learning rate:{}, collaboration rate:{}, creativity:{}, layer:{}'\
			.format(self.id, self.learning,self.collaboration, self.creativity, self.hierarchy)

	def know_info(self):
		plt.hist(self.knowledge, bins=10)
		plt.title('Id:{}_Knowledge Distribution'.format(self.id))
		plt.show()
		return {'dist':(self.mu, self.std), 'content':self.knowledge}

	def reaction(self, perceived_perf, hetero=1, M=20):
		if perceived_perf == 0: #현재 자신의 효용이 감소했다고 판단 -> 현재 자신의 행동을 수정
			n = np.random.randint(0,M)
			m = np.random.randint(0,(Agent.number_of_agent)/2)
			know = self.knowledge[n]

			# 지식내용 변경
			if np.random.rand() >= 0.5:
				self.knowledge[n] = np.random.randint(-M,M)

			# 지식구조 변경
			else:
				for x in range(0,len(self.knowledge)):

					x=np.random.randint(1, 6)

					if np.random.rand() >= 0.5:
						self.knowledge = [k+x for k in self.knowledge]
					else:
						self.knowledge = [k-x for k in self.knowledge]

			#네트워크 변경
			if Agent.adj[self.id][m] == 0:
				Agent.adj[self.id][m] = 1
				Agent.adj[m][self.id] = 1
			else:
				Agent.adj[self.id][m] = 0
				Agent.adj[m][self.id] = 0

		elif hetero == 0: #현재 조직내 동질성이 감소했다고 판단
			n = np.random.randint(0,M)
			m = np.random.randint(0,(Agent.number_of_agent)/2)
			know = self.knowledge[n]

			# 지식내용 변경
			if np.random.rand() >= 0.5:
				self.knowledge[n] = np.random.randint(-M,M)

			# 지식구조 변경
			else:
				for x in range(0,len(self.knowledge)):

					x=np.random.randint(1, 6)

					if np.random.rand() >= 0.5:
						self.knowledge = [k+x for k in self.knowledge]
					else:
						self.knowledge = [k-x for k in self.knowledge]

			#네트워크 변경
			if Agent.adj[self.id][m] == 0:
				Agent.adj[self.id][m] = 1
				Agent.adj[m][self.id] = 1
			else:
				Agent.adj[self.id][m] = 0
				Agent.adj[m][self.id] = 0


#Functions

## initialize

## Construct Schema Distribution
def make_schema(N,M):
	mu = np.random.randint(-10,10)
	#schema = [x for x in np.random.randint(0,1, M*N)]
	schema = [round(x) for x in np.random.normal(mu,10,M*N)]

	return schema

#Organizational knowledge distribution
def organizational_know(agent_list,N=50):

	organizational_know = []

	for a in agent_list:
		organizational_know = a.knowledge + organizational_know

	return organizational_know

# make agent
def make_agent(N,M):

	agent_list = []

	for i in range(0,N):
		learning = np.random.randint(1,4)/10
		collaboration = np.random.randint(1,4)/10
		creativity = np.random.randint(1,4)/10
		mu = np.random.randint(-M,M)
		std = np.random.randint(0,10)
		agent_list.append(Agent(i,learning,collaboration,creativity,M,mu,std))

	for i in range(0,N):
		for j in range(i+1,N):
			if np.random.rand() > 0.8:
				Agent.adj[i][j] = 1
				Agent.adj[j][i] = 1
			else:
				Agent.adj[i][j] = 0
				Agent.adj[j][i] = 0

	return agent_list

## Performance calculation
def performance_calculation(N, schema, org_know):

	d = 0
	k = (1/N)

	schema = collections.Counter(schema)
	schema = sorted(schema.items(), key=operator.itemgetter(0))

	org_know = collections.Counter(org_know)
	org_know = sorted(org_know.items(), key=operator.itemgetter(0))

	for i in range(0,len(schema)):
		for j in range(0,len(org_know)):
			if schema[i][0] == org_know[j][0]:
				d = d + abs(schema[i][1] - org_know[j][1])**2

	return d**k

## learning
def learning(agent, agent_list, M):

	i = np.random.randint(0,M)

	while (Agent.adj[agent.id][i] == 1) and (id != i+1):
		i = np.random.randint(0, M)

	n = np.random.randint(0, len(agent.knowledge))

	#hierarchy에 의한 학습 효과의 차이
	if agent_list[i].hierarchy > agent.hierarchy:
		if agent.learning*2 > np.random.rand():
			agent.knowledge[n] = copy.copy(agent_list[i].knowledge[n])
	elif agent_list[i].hierarchy == agent.hierarchy:
		if agent.learning > np.random.rand():
			agent.knowledge[n] = copy.copy(agent_list[i].knowledge[n])
	else:
		if agent.learning/2 > np.random.rand():
			agent.knowledge[n] = copy.copy(agent_list[i].knowledge[n])

	return 0

def collaboration(agent, agent_list,M):

	member = [i for i,x in enumerate(list(Agent.adj[agent.id])) if x == 1 ]

	m = member[np.random.randint(0,len(member))]

	n = np.random.randint(0, M)

	conccurence = agent_list[m].knowledge[n]

	for i in member:
		if agent_list[i].collaboration >= np.random.rand():
			agent_list[i].knowledge[n] = conccurence * random.randint(50,100)/100
		else:
			agent_list[i].knowledge[n] = (conccurence + agent_list[i].knowledge[n])/2

	return 0


#No collaboration, No learning (Baseline model)
def basic_model(agent_list, schema, initial_org_know, col = model_type, max_iter=100, N=50, M=20):

	schema_accu = pd.Series(schema)
	org_know_var = []

	for i in range(0, max_iter):

		if i == 0:
			# 성과어레이 초기화
			performance = pd.Series([])
			performance[0] = 1.0
			'''
			sns.distplot(org_know, bins = 20, color='grey', label='organiztional knowledge')
			sns.distplot(schema, bins = 20, color='blue', label='Schema')
			plt.legend()
			plt.title('t={}'.format(i+1))
			plt.show()
			'''
			org_know_accu = pd.Series(initial_org_know)
			org_know_var.append(np.var(initial_org_know))
			continue

		elif i % 20 == 0:
			'''
			sns.distplot(org_know, bins = 20, color='grey', label='organiztional knowledge')
			sns.distplot(schema, bins = 20, color='blue', label='Schema')
			plt.legend()
			plt.title('t={}'.format(i+1))
			plt.show()
			'''

		elif i == (max_iter - 1):
			'''
			sns.distplot(org_know, bins = 20, color='grey', label='organiztional knowledge')
			sns.distplot(schema, bins = 20, color='blue', label='Schema')
			plt.legend()
			plt.title('t={}'.format(i+1))
			plt.show()
			'''

		org_know = organizational_know(agent_list, N)
		org_know_var.append(np.var(org_know))
		org_know_accu = pd.concat([org_know_accu, pd.Series(org_know)], axis=1, ignore_index=True)
		performance[i] = performance_calculation(N, schema, org_know)

		for a in agent_list:

			if col =='on':
				collaboration(a, agent_list, M)
				learning(a, agent_list, M)

			else:
				learning(a, agent_list, M)

			if i == 0:
				a.reaction(0,M)
			elif performance[i] - performance[i-1] > 0:
				a.reaction(0,M)
			#elif org_know_var[i] - org_know_var[i-1] > 0:
			#	a.reaction(1,M)
			#else:
			#	a.reaction(1,M)


	return (performance, org_know_accu)

#%% Initializing

performance = pd.DataFrame([])
accu_know = []


schema = make_schema(N, M)

#Implementation
for i in range(0,ITER):
	schema = make_schema(N, M)

	Agent.adj = np.ones((N,N))
	Agent.number_of_agent = 0

	agent_list = make_agent(N, M)

	org_know = organizational_know(agent_list, N)

	temp_rlt = basic_model(agent_list,schema,org_know,max_iter=max_iter,col = model_type)

	performance = pd.concat([performance, temp_rlt[0]], ignore_index = True, axis=1 )

	accu_know.append(temp_rlt[1])

	print("{} iteraction...".format(i+1))

# 자료형 확인
#print(type(accu_know), type(performance))

# 평균 조직 지식, 평균 퍼포먼스 값 데이터셋 구축

avg_know = pd.DataFrame(np.zeros((M*N,max_iter)))

for i in range(0,ITER):

	avg_know = avg_know + accu_know[i]

avg_know = avg_know/ITER
avg_var = avg_know.var(axis=0)**0.5


avg_perf = pd.DataFrame(performance.mean(axis=1))


# 데이터 값 확인
#print(avg_know.head(5),avg_perf.head(5), sep='\n')

#%%
print(accu_know[0])
#%% 지식 분포 변화 추이
tu=time.localtime()[3:5]

for col in avg_know.columns:

	if col == 0:
		sns.distplot(avg_know[col], bins=20, color='grey', label='organiztional knowledge')
		sns.distplot(schema, bins=20, color='blue', label='Schema')
		plt.legend()
		plt.xlabel('knowledge landscape')
		plt.ylabel('Weight of knowledge')
		plt.title('t={}'.format(col + 1))
		plt.savefig("result\\know_{}_t_{}_{}.png".format(model_type,col+1,str(tu)))
		plt.show()

	elif col%20 == 0:
		sns.distplot(avg_know[col], bins = 20, color='grey', label='organiztional knowledge')
		sns.distplot(schema, bins = 20, color='blue', label='Schema')
		plt.legend()
		plt.xlabel('knowledge landscape')
		plt.ylabel('Weight of knowledge')
		plt.title('t={}'.format(col +1))
		plt.savefig("result\\know_{}_t_{}_{}.png".format(model_type, col+1, str(tu)))
		plt.show()

	elif col == max_iter-1:
		sns.distplot(avg_know[col], bins = 20, color='grey', label='organiztional knowledge')
		sns.distplot(schema, bins = 20, color='blue', label='Schema')
		plt.legend()
		plt.xlabel('knowledge landscape')
		plt.ylabel('Weight of knowledge')
		plt.title('t={}'.format(col +1))
		plt.savefig("result\\know_{}_t_{}_{}.png".format(model_type, col+1, str(tu)))
		plt.show()

	else:
		continue

#%% 퍼포먼스 추이

plt.plot(performance.index, 1/(avg_perf+1), alpha=2, marker='d',markersize=10,markeredgecolor='b',antialiased=True, color='black')
plt.ylabel("Avg. PERF.")
plt.xlabel("t")
plt.ylim((0, 1.05))
plt.xlim((1,100))
plt.savefig("result\\perf_{}_{}.png".format(model_type, str(tu)))
plt.show()


#%%분산 추이

plt.plot(performance.index, avg_var, alpha=2, marker='d',markersize=5,markeredgecolor='b',antialiased=True, color='black')
plt.ylabel("Avg. VAR.")
plt.xlabel("t")
plt.ylabel('Avg. Var')
plt.xlim((1,100))
plt.savefig("result\\var_{}_{}.png".format(model_type, str(tu)))
plt.show()

#%%
avg_var_cohe = copy.copy(avg_var)
avg_perf_cohe = copy.copy(avg_perf)

#%%
avg_var_insul = copy.copy(avg_var)
avg_perf_insul = copy.copy(avg_perf)

#%%
avg_var_homo_low= copy.copy(avg_var)
avg_perf_homo_low = copy.copy(avg_perf)
#%%
avg_var_homo_high = copy.copy(avg_var)
avg_perf_homo_high = copy.copy(avg_perf)
#%%
avg_var_norm = copy.copy(avg_var)
avg_perf_norm = copy.copy(avg_perf)
#%%
avg_var_lead = copy.copy(avg_var)
avg_perf_lead = copy.copy(avg_perf)

#%%
li = [avg_perf_cohe,avg_perf_insul,avg_perf_homo_high,avg_perf_homo_low,avg_perf_norm,avg_perf_lead]
name = ['cohe','insul','homo_high','homo_low','norm','lead']
s = pd.Series(li, index = name)

for i in s.index:
	s[i].to_csv('result_{}.csv'.format(i))
#%%
li = [avg_var_cohe,avg_var_insul,avg_var_homo_high,avg_var_homo_low,avg_var_norm,avg_var_lead]
name = ['cohe','insul','homo_high','homo_low','norm','lead']
s = pd.Series(li, index = name)

for i in s.index:
	s[i].to_csv('var_{}.csv'.format(i))


#%%
avg_var.to_csv('var.csv')
avg_perf.to_csv('perf.csv')
#%%
avg_perf
