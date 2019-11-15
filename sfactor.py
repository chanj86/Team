# Switching factors
'''
1. Collision
- 이질적인 지식들 간의 의도적인 충돌

2. Delay
- 행위자들이 상호작용하는 대안(지식)을 수용하는데 시간차를 두고 수용함

3. Retrieval
- 과거의 지식을 폐기하지 않고 저장해놨다가 다시 사용
'''

# import modules

import pandas as pd
import numpy as np
import math
import collections
import random

#%%
#(1) knowledge collision

def find_max_hetero(AGENT, agent_list, agent):

	max_hetero = [0,0]

	for i,a in enumerate(agent_list):

		d = abs(agent.mu - a.mu)

		if d > max_hetero[1]:

			max_hetero[0] = i
			max_hetero[1] = d

		else:
			continue

	return max_hetero[0]

def knowledge_collision(AGENT, agent_list):

	for a in agent_list:

		mate = agent_list[find_max_hetero(AGENT, agent_list, a)]
		drift = np.mean(mate.knw) - np.mean(a.knw)
		# print(a.id,':',n,':',drift)
		temp = []

		# 모든 개별지식을 drift+noise 만큼 이동
		for k in a.knw:
			if np.random.rand() < 0.05:

				e = np.random.uniform(0.5, 1.0)

				if k * e > 100 or k * e < -100:  # 지나치게 스코프를 벗어나는 지식은 배제함
					temp.append(round(np.random.randint(-50, 50)))
				else:
					temp.append(round(k * e))

			else:
				temp.append(k)

		a.knw = temp

#%%
# (2) delay

def knowledge_delay():

	return  np.random.uniform(0.05,0.1)
#%%
# (3) retrieval

def knowledge_store(AGENT, agent_list, knw_storage, iter):

	a = random.choice(agent_list)

	knw_storage[iter] = a.knw

	return a.knw

def knowledge_retrieval(AGENT, agent_list, knw_storage,iter):

	for a in agent_list:

		if iter < 50:

			return None

		elif iter >= 50 and np.random.rand() < 0.1:

			i = np.random.randint(0,iter)

			#드리프트항
			drift =  np.mean(list(knw_storage[i])) - np.mean(a.knw)
			#print(a.id,':',n,':',drift)
			temp = []

			# 모든 개별지식을 drift+noise 만큼 이동
			for k in a.knw:
				e = np.random.uniform(0.3,0.8)
				if k*e > 50 or k*e < -50: #지나치게 스코프를 벗어나는 지식은 배제함
					temp.append(round(np.random.randint(-10,10)))
				else:
					temp.append(round(k*e))

			a.knw = temp

			return None

		else:

			continue

	return None




#%%

#%%


