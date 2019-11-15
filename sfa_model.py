# %% 20190905
## Agent Layer
## Namjun CHA, Seoul National Univiersity, 2019
## This module includes function and parameters of 'agent'.

# %% 모듈

import pandas as pd
import numpy as np
import math
import seaborn as sns
from matplotlib import pyplot as plt
import collections
import operator
import time
import sfactor as sf


#%% 행위자 클래스 선언


class AGENT:

	#조직수준변수
	n = 0 #행위자 수
	k = 100 # 지식의 개수
	knw_opt = [int(x*10) for x in np.random.normal(0,0.5,k)]# 최적 지식
	knw_org = [] # 조직 지식
	perf_org = 0 # 조직 성과
	adj = [] # 네트워크 행렬

	def __init__(self,id,learn = 0.3, div=50):

		AGENT.n += 1 # 행위자 수 1 증가

		self.id = id # 행위자 id
		self.div = div
		self.learn = learn

		#행위자 효용함수
		self.k = np.random.uniform(0.1,1.0)
		self.alpha = np.random.uniform(0.5,1.0)
		self.perf = 1
		self.util = self.k * (self.perf**self.alpha) * (AGENT.perf_org**(1-self.alpha))

		#행위자 지식 계층
		self.mu = np.random.randint(10,self.div)
		self.sigma = np.random.uniform(2,self.div/10)
		self.knw = [x for x in np.random.randint(self.mu-round(self.sigma**2),self.mu+round(self.sigma**2),AGENT.k)]

		#인접한 행위자 리스트 (다른 행위자의 id가 기록됨)
		self.neighbor = []
		self.rank = [0,0]

	# 행위자의 표현형 (id 출력)
	def __repr__(self):

		return 'id:{}'.format(self.id)

	def show_info(self,type='org'):

		bins = 50

		if type == 'org':
			'''
			sns.distplot(AGENT.knw_opt, bins=30, color='grey', label='knw_opt')
			sns.distplot(AGENT.knw_org, bins=30, color='blue', label='knw_org')
			plt.legend()
			plt.xlabel('knowledge landscape')
			plt.ylabel('Weight of knowledge')
			plt.title('opt-{}'.format(type))
			plt.show()
			'''
			sns.kdeplot(AGENT.knw_org, color='grey')
			sns.kdeplot(AGENT.knw_opt, color='red')
			plt.title('organizational knowledge')
			plt.show()

		else:
			sns.distplot(AGENT.knw_opt, bins=30, color='grey', label='knw_opt')
			sns.distplot(self.knw, bins=30, color='blue', label='knw_ind')
			plt.legend()
			plt.xlabel('knowledge landscape')
			plt.ylabel('Weight of knowledge')
			plt.title('opt-{}'.format(type))
			plt.show()

		return AGENT.perf_org



#%% 함수

#1-1.조직 지식 업데이트
# 행위자의 지식을 합쳐서 조직지식을 업데이트함
def update_knw_org(AGENT, agent_list):

	n = len(agent_list)
	knw_org = []

	for a in agent_list:
		knw_org = knw_org + a.knw

	AGENT.knw_org = knw_org

	return knw_org

#1-2. 행위자 효용 업데이트
def update_utility(AGENT, agent_list):

	for a in agent_list:

		a.util = a.k*((a.perf+1)**a.alpha)*((AGENT.perf_org+1)**(1-a.alpha))

		if a.util == 0:
			a.k*((1)**a.alpha)*((1)**(1-a.alpha))

	return None

#1-3. 행위자 지식 평균/편차 업데이트
def update_mu_sigma(AGENT, agent_list):
	for a in agent_list:
		a.mean = np.mean(a.knw)
		a.sigma = np.std(a.knw)

	return (a.mean, a.sigma)


#2. 성과 측정
# 최적 지식 대비 다른 지식의 성과를 측정함
def update_perf_ind(AGENT, agent_list, N):

	for a in agent_list:

		knw_opt = collections.Counter(AGENT.knw_opt)
		knw = collections.Counter(a.knw)

		knw_opt_key = knw_opt.keys()
		knw_key = knw.keys()

		d = 0 # 두 지식분포간의 격차

		knw_opt = sorted(knw_opt.items(), key=operator.itemgetter(0))
		knw = sorted(knw.items(), key=operator.itemgetter(0))

		for i in range(0,len(knw_opt)):

			if knw_opt[i][0] in knw_key:

				for j in range(0,len(knw)):

					if knw_opt[i][0] == knw[j][0]:

						d = d + abs(knw_opt[i][1] - knw[j][1])

		perf = d / 100
		a.perf = perf  #성과의 역수

	return perf

def update_perf_org(AGENT, agent_list, N):

	knw_opt = collections.Counter(AGENT.knw_opt)
	knw_org = collections.Counter(AGENT.knw_org)

	knw_opt_key = knw_opt.keys()
	knw_org_key = knw_org.keys()

	d = 0 # 두 지식분포간의 격차

	knw_opt = sorted(knw_opt.items(), key=operator.itemgetter(0))
	knw_org = sorted(knw_org.items(), key=operator.itemgetter(0))

	for i in range(0,len(knw_opt)):

		if knw_opt[i][0] in knw_org_key:

			for j in range(0,len(knw_org)):

				if knw_opt[i][0] == knw_org[j][0]:

					d = d + abs(knw_opt[i][1] - round(knw_org[j][1]/N))

		else:

			d = d + knw_opt[i][1]

	perf = d
	AGENT.perf_org = perf # 성과의 역수

	return perf

#3-1. 행위자 네트워크 만들기 (랜덤네트워크)
def make_network(AGENT, agent_list, density = 0.3):
	AGENT.adj = np.zeros((len(agent_list), len(agent_list)))

	for i in range(len(agent_list)):
		for j in range(len(agent_list)):
			if i == j:
				AGENT.adj[i][j] = 1
				AGENT.adj[j][i] = 1

			elif np.random.rand() < density:
				AGENT.adj[i][j] = 1
				AGENT.adj[j][i] = 1

			else:
				continue

	return AGENT.adj

def update_network(AGENT, agent_list):

	#교체할 링크 선택
	for a in agent_list:

		link = np.random.choice(a.neighbor)

		AGENT.adj [a.id][link] = AGENT.adj[link][a.id] = 0

		connect = 0

		while connect ==1 :

			choice = np.ranmdom.choice(list(range(0,len(agent_list))))

			if agent_list[choice].util > a.util:

				AGENT.adj[a.id][choice] = AGENT.adj[choice][a.id] = 1

				connect = 1

			else:
				continue

	return AGENT.adj

#3-2. 인접 행위자 리스트 업데이트
def find_neighborhood(AGENT, agent_list):

	for i in range(len(agent_list)):
		for j in range(len(agent_list)):

			if AGENT.adj[i][j] == 1 and i!=j:
				agent_list[i].neighbor.append(j)

	return None

#4-1 행위자 효용 랭크
def update_rank(AGENT, agent_list):

	for a in agent_list:

		temp_dict = {a.id: a.perf}

		for i in a.neighbor:

			temp_dict[i] = agent_list[i].perf

		sorted_dict = sorted(temp_dict.items(), key=lambda t: t[1], reverse=True)

		a.rank[0] = sorted_dict.index((a.id, a.perf)) + 1
		a.rank[1] = len(a.neighbor)

	return None

#4-2 평균 효용 계산

def average_util(AGENT, agent_list):

	sum_util = 0
	i = 0
	for a in agent_list:
		sum_util += a.util
		i += 1

	return sum_util/i


#5. 지식 상호작용
# drift항과  noise항을 활용
def knowledge_iteraction(AGENT, agent_list):

	learn = agent_list[0].learn

	for a in agent_list:

		for n in a.neighbor:

			mate = agent_list[n]

			if a.util < mate.util and (a.rank[0]/a.rank[1]) > 0.5 and np.random.rand() < learn : #상대방의 효용이 더 높을 때 그리고 하위 50%의 성과를 지닐때

				#드리프트항
				drift =  np.mean(mate.knw) - np.mean(a.knw)
				#print(a.id,':',n,':',drift)
				temp = []

				# 모든 개별지식을 drift+noise 만큼 이동
				for k in a.knw:
					e = np.random.uniform(0.5,1.0)
					if k*e > 50 or k*e < -50: #지나치게 스코프를 벗어나는 지식은 배제함
						temp.append(round(np.random.randint(-10,10)))
					else:
						temp.append(round(k*e))

				a.knw = temp

			else:

				continue

	return None

def knowledge_iteraction_delay(AGENT, agent_list):

	learn = agent_list[0].learn

	for a in agent_list:

		for n in a.neighbor:

			mate = agent_list[n]

			if a.util < mate.util and (a.rank[0]/a.rank[1]) > 0.5 and np.random.rand() < learn: #상대방의 효용이 더 높을 때 그리고 하위 50%의 성과를 지닐때

				#드리프트항
				drift =  np.mean(mate.knw) - np.mean(a.knw)
				#print(a.id,':',n,':',drift)
				temp = []

				# 모든 개별지식을 drift+noise 만큼 이동
				for k in a.knw:
					e = np.random.uniform(0.5,1.0)
					if k*e > 50 or k*e < -50: #지나치게 스코프를 벗어나는 지식은 배제함
						temp.append(round(np.random.randint(-10,10)))
					else:
						temp.append(round(k*e))

				a.knw = temp

			else:

				continue

	return None

#6. 지식 돌연변이 (자생적 변화 및 창의성)
def knowledge_mutation(AGENT, agent_list):

	#랜덤 에이전트 선택 (전체의 10%)

	for a in agent_list:

		if a.rank[0]/a.rank[1] >= 0.9:

			for k,i in zip(a.knw, range(len(a.knw))):

				e = np.random.randint(-10, 10)

				if -50 <= k+e <= 50:
					a.knw[i] = k+e
				else:
					continue

	return None

#7. 지식 분포를 정규분포화
def pdf(x, mu, sigma):
	a = 1 / (sigma * np.sqrt(2 * np.pi))
	b = -1 / (2 * sigma ** 2)

	return a * np.exp(b * (x - mu) ** 2)


def normal_distribution(AGENT, agent_list):

	x = np.linspace(-50, 50, 100)
	# 행위자들 분포
	'''
	for a in agent_list:

		knw = a.knw

		mu, sigma = np.mean(knw), np.std(knw)

		plt.plot(x, pdf(x, mu, sigma + 1), color='.75')
	'''
	# 최적 지식 분포
	plt.plot(x, pdf(x, np.mean(AGENT.knw_opt), np.std(AGENT.knw_opt)), color='red')
	# 조직 지식 분포
	plt.plot(x, pdf(x, np.mean(AGENT.knw_org), np.std(AGENT.knw_org)))
	# 플롯 출력
	plt.show()

	return None

#8. 평균과 분산의 scatter plot
def plot_mu_sigma(AGENT, agent_list):
	mu = []
	sigma = []

	for a in agent_list:
		mu.append(a.mu)
		sigma.append(a.sigma**2)
	sns.kdeplot(mu,sigma,cmap='Blues',shade=True,shade_lowest=False)
	plt.ylim(-100,300)
	plt.xlim(0,50)
	plt.show()
	#plt.scatter(mu,sigma,color='0.25')
	#plt.show()

	return None

#9. 지식의 편향성 분석 (bias)

def knowledge_bias(AGENT, agent_list):

	total_d = 0

	for a in agent_list:
		for b in agent_list:

			d = 0.5*np.sqrt((a.mu - b.mu)**2+(a.sigma**2-b.sigma**2)**2)
			total_d += d

	return total_d/len(agent_list)


#%% 초기화

## 전역변수

iter = 30  #실험당 반복수
EXP_ITER = 1000 #전체 실험
N = 50 # 에이전트의 수

knw_storage = pd.DataFrame([], columns = list(range(0,iter)),index = list(range(0,100)))
sfa_column = ['id','year', 'perf','div','learn','div^2','learn^2','div_learn','group']
sfa_df = pd.DataFrame([],columns = sfa_column, index=range(0,EXP_ITER))

#group 번호
'''
group=1: refer
group=2: know collsion + delay
group=3: know delay + retrieval
group=4: know collision + retrieval
group=5: All
group=6: know collision
group=7: know delay
group=8: know retrieval
'''
group = 4

for ITER in range(0,EXP_ITER):

	print('##{}th iteration.'.format(ITER))
	# 무작위로 다양성과 학습능력 조절
	diversity = np.random.uniform(30, 100)
	learn = np.random.uniform(0.1, 0.99)

	#Step0) 에이전트 생성
	agent_list = []
	for i in range(0,N):
		agent_list.append(AGENT(i,learn, diversity))

	## Step1) 모형 초기화
	# 행위자 네트워크 생성
	make_network(AGENT, agent_list)
	find_neighborhood(AGENT, agent_list)

	# 조직 지식 업데이트
	update_knw_org(AGENT, agent_list)

	# 조직 지식의 성과 업데이트 (perf_org)
	update_perf_org(AGENT, agent_list, N)

	# 개별 지식의 성과 업데이트 (perf)
	update_perf_ind(AGENT, agent_list, N)

	# 개별 지식 평균, 편차 업데이트
	update_mu_sigma(AGENT, agent_list)

	# 개별 행위자의 효용 업데이트
	update_utility(AGENT, agent_list)

	# 개별 행위자의 랭크 업데이트
	update_rank(AGENT, agent_list)

	## Step2) 초기화 점검
	# 행위자 수
	if len(agent_list) == N:
		print('##{} Agents are created.'.format(N))
	else:
		raise ValueError('number of agent is invalid')

	## Step2)기본모형 실행
	#agent_list[0].show_info()
	#normal_distribution(AGENT, agent_list)
	#plot_mu_sigma(AGENT, agent_list)

	for i in range(0,iter+1):
		knowledge_iteraction(AGENT, agent_list,)
		#knowledge_iteraction_delay(AGENT, agent_list)
		knowledge_mutation(AGENT, agent_list)
		update_knw_org(AGENT, agent_list)
		update_perf_ind(AGENT, agent_list, N)
		update_perf_org(AGENT, agent_list, N)
		update_utility(AGENT, agent_list)
		update_network(AGENT, agent_list)
		update_mu_sigma(AGENT, agent_list)
		sf.knowledge_collision(AGENT,agent_list)
		sf.knowledge_store(AGENT, agent_list, knw_storage, i)
		sf.knowledge_retrieval(AGENT, agent_list, knw_storage, i)

		#중간과정 확인
		'''
		if i%int(iter/3) == 0:
			agent_list[0].show_info()
			print('perf_org:', 1/AGENT.perf_org, '\nOrg Know:', AGENT.knw_org, '\nOptimum Know:', AGENT.knw_opt)
			plot_mu_sigma(AGENT,agent_list)
			#normal_distribution(AGENT, agent_list)
		'''

	## Step3) 데이터 업데이트
	print("##데이터 업데이트\n\n")
	sfa_df['perf'][ITER] = AGENT.perf_org
	sfa_df['div'][ITER] = diversity
	sfa_df['learn'][ITER] = learn

print('###데이터저장###')
sfa_df['id'] = sfa_df.index + 1
sfa_df['year'] = 1
sfa_df['div^2'] = sfa_df['div']**2
sfa_df['learn^2'] = sfa_df['learn']**2
sfa_df['div_learn'] = sfa_df['div']*sfa_df['learn']
sfa_df['group'] = group

# 데이터 백업
print(sfa_df.head(5))

#현재시간불러오기

now = time.gmtime(time.time())
time_str = '{}{}_{}{}'.format(now.tm_mon,now.tm_mday,now.tm_hour,now.tm_min)

sfa_df.to_csv('result\\sfa_df_{}_{}.csv'.format(group,time_str), header=True)


