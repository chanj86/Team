#%% 20190905
## Agent Layer
## Namjun CHA, Seoul National Univiersity, 2019
## This module includes function and parameters of 'agent'.

#%% 1.Import modules

import pandas as pd
import numpy as np
import math
import seaborn as sns
from matplotlib import pyplot as plt
import collections
#%% 2.Declration of "AGENT" class

class AGENT:

	#class variables
	network = []
	number_of_agent = 0
	p_org = 0.5 #initial condition
	knw_opt_raw = [round(x * 10) for x in np.random.normal(0, 1, 100)]
	knw_opt = collections.Counter(knw_opt_raw)
	knw_org_raw = []
	knw_org = {}

	#initialization function
	def __init__(self,id,alpha,k):

		# Increasing number of agent
		AGENT.number_of_agent += 1

		self.id = id
		self.alpha = alpha
		self.k = k

		self.status = 0 #0: sleep, 1: active

		self.rand_num = np.random.rand()

		'''
		if self.rand_num < 0.2:
			self.mu = np.random.randint(-50,50)
		elif self.rand_num < 0.4:
			self.mu = np.random.randint(50, 100)
		elif self.rand_num < 0.6:
			self.mu = np.random.randint(-100, -50)
		else:
			self.mu = np.random.randint(-100,100)
		'''
		self.mu = np.random.uniform(-10,10)
		self.var = np.random.uniform(0,50,1)
		self.rank = [0,0]
		self.neighbor = []

		#initial update
		self.knw_raw = [round(x) for x in np.random.normal(self.mu, self.var, 100)]
		self.knw_dict = self.Update_Knowledge()

		self.p_ind = self.Calculate_Performance(self.knw_dict, self.knw_opt)+1
		self.util = self.Calculate_Util()

	def __repr__(self):
		info = {'id':self.id, 'alpha':self.alpha, 'sensitivity(k)':self.k}
		return '{}'.format(info)

	#Information of agent
	def Show_Info(self):
		print('id:{}\nutility:{} (alpha={},k={})'.format(self.id, self.util,self.alpha, self.k))
		print('knowledge: mean={}, var={}, performance: {}'.format(self.mu, self.var, self.p_ind))
		print('performance rank: {}'.format(self.rank))

		sns.distplot(self.knw_opt_raw, bins=40, color='grey', label='knw_opt')
		sns.distplot(self.knw_raw, bins=20, color='blue', label='knw_ind')
		plt.legend()
		plt.xlabel('knowledge landscape')
		plt.ylabel('Weight of knowledge')
		plt.title('id={}'.format(self.id))
		plt.show()

		return None

	# update knw_raw --> knw_dict (frequency)
	def Update_Knowledge(self):

		self.knw_dict = collections.Counter(self.knw_raw)

		return self.knw_dict

	# update mean and var
	def Update_Mu_Var(self):

		self.mu = np.mean(self.knw_raw)
		self.var = np.var(self.knw_raw)

	# Calculating Utility of agent
	def Calculate_Util(self):
		# parameters
		alpha = self.alpha
		k = self.k
		p_ind = self.p_ind

		# calculate utility of agent
		self.util = k * (p_ind ** alpha) * (self.p_org) ** (1 - alpha)

		return self.util

	# Calculating performance of agent
	def Calculate_Performance(self,knw, knw_opt):
		# prameters
		d = 0  # initial distance

		# distance between agent knowledge and optimum knowledge
		for key in knw_opt.keys():
			d += abs(knw_opt[key] - knw[key])

		perf = abs(1-(d / len(knw_opt.keys())))

		return perf

	# Network construction (initial configuration only)
	def Initial_Network(self):
		size = self.number_of_agent
		adj = np.zeros((size, size))
		for i in range(0, size):
			for j in range(0, size):
				if i == j:
					adj[i][j] = 1

				elif np.random.rand() < 0.99:
					adj[i][j] = 1
					adj[j][i] = 1

				else:
					adj[i][j] = 0
					adj[j][i] = 0

		AGENT.network = adj

		return AGENT.network

	# Finding neihobor of agent (under 1 link)
	def Find_Neighbor(self):

		col = self.network[self.id]

		for i in range(0,len(col)):
			if col[i] == 1:

				self.neighbor.append(i)

		return self.neighbor

	# Calculate rank of performance (in neighborhood)
	def Calculate_Rank(self, agent_list):

		temp_dict = {self.id:self.p_ind}

		for i in self.neighbor:
			temp_dict[i] = agent_list[i].p_ind

		sorted_dict = sorted(temp_dict.items(), key=lambda t: t[1], reverse=True)

		self.rank[0] = sorted_dict.index((self.id, self.p_ind))+1
		self.rank[1] = len(self.neighbor)

		return (self.rank, len(self.neighbor))


################################################################################################
## 3.Functions
#(1) Calculating organizational knoweldge performance
def Calculating_Performance_Org(AGENT, knw_opt_raw,knw_opt, agent_list, ITER=0):

	#merging individual knowledge
	knw_sum = []

	for i in range(len(agent_list)):
		knw_sum = knw_sum + agent_list[i].knw_raw

	knw_dict_sum = collections.Counter(knw_sum)

	for i in knw_dict_sum.keys():
		knw_dict_sum[i] = knw_dict_sum[i]/len(agent_list)

	AGENT.knw_org_raw = knw_sum
	AGENT.knw_org = knw_dict_sum

	# plotting distiributions
	if (ITER+1)%100 == 0:
		sns.distplot(knw_opt_raw, bins=10, color='grey', label='knw_opt')
		sns.distplot(knw_sum, bins=10, color='blue', label='knw_org')
		plt.legend()
		plt.xlabel('knowledge landscape')
		plt.ylabel('Weight of knowledge')
		plt.title('performance of organizational knowledge(t={})'.format(ITER+1))
		plt.show()
	else:
		print('ITER=',ITER)

	# calculating distance between optimum knowledge and organizational knowledge
	d = 0  # initial distance

	for key in knw_opt.keys():
		d += knw_opt[key] - knw_dict_sum[key]

	perf = abs(1 - (d / 100))+1

	return perf

#(2) Knowledge interaction between two agents
def Knowledge_Interaction(agent_list, agent, l_ratio = 0.7): #l_ratio: probability to learn from other agent
	'''
	for neighbor in agent.neighbor:

		agent_mate = agent_list[neighbor]


		if (np.random.rand(0,1) < l_ratio) and (agent_mate.util > agent.util): # when utility of agent_mate is higher


			for i in range(0,len(agent.knw_raw)):

				if np.random.rand() < l_ratio:
					agent.knw_raw[i] = agent_mate.knw_raw[i]

				else:
					continue
		else:
			continue
	'''

	for n in agent.neighbor:

		agent_mate = agent_list[n]

		if agent_mate.util > agent.util:

			gap = np.mean(agent_mate.knw_raw) - np.mean(agent.knw_raw)

			noise = np.random.randint(-1, 1)

			delta = round(gap+noise)

			# moving knowledge distribution of agent
			agent.knw_raw = [x + delta for x in agent.knw_raw]

		else:
			continue

	return agent.knw_raw

#(3) Update network of agent
def Update_Network(AGENT):
	for i in range(AGENT.number_of_agent):
		for j in range(AGENT.number_of_agent):
			if (AGENT.network[i][j] == 1) and (np.random.rand() > 0.1):
				AGENT.network[i][j] = 0
				AGENT.network[j][i] = 0
			else:
				continue

	return AGENT.network

# (4) Calculate avearge utility of agent
def Calculate_Avg_Util(agent_list,N=100):
	u = 0
	for agent in agent_list:
		u += agent.util
	u = u / N

	return u
#%% Experiment
###########################################################
#################### Implementation #######################
###########################################################

####################
## 1.Initializing ##
####################

# parameters
N = 50 #number of agent
ITER = 300 #number of iteration
SEED = 1 #seed number of random functions
np.random.seed(SEED)

# Agent instances
agent_list = []
for i in range(N):
	agent_list.append(AGENT(id=i,alpha=np.random.uniform(0.9,1.0),k=np.random.uniform(0.3,1.0)))

# Agent newtork
agent_list[0].Initial_Network()

for i in range(N):
	agent_list[i].Find_Neighbor()

#  Calculating organizational performance
AGENT.p_org = Calculating_Performance_Org(AGENT, AGENT.knw_opt_raw, AGENT.knw_opt,agent_list)

# Calculate relative performance (rank)
for i in range(N):
	agent_list[i].Calculate_Rank(agent_list)

#%%###################
## 2.Implementation ##
######################

##########
# (1) selecting activated agent (status = 1)
# agent who belongs to under 50% of utility ==> Activated
# otherwise ==> sleep
# this is for reducing comuptational time
for agent in agent_list:
	if agent.rank[0]/agent.rank[1] < 0.5: #under 50%
		agent.status = 1 #activating
	else:
		agent.status = 0 #sleep

# dataframe of organizational performance
p_org_df = [AGENT.p_org]

# dataframe of mean and deviation of organizational knowledge
mu_org_df = [np.mean(AGENT.knw_opt_raw)]
var_org_df = [np.var(AGENT.knw_opt_raw)]
avg_util_df = [Calculate_Avg_Util(agent_list, 100)]

###########
# (2) interaction between agents

for iter in range(ITER):

	print(agent_list[0].knw_dict)
	# Knoweldge Interaction
	for agent in agent_list:
		if agent.status == 1:
			agent.knw_raw = Knowledge_Interaction(agent_list, agent)
			agent.Update_Knowledge()
		else:
			continue

###########
#(3) Update information
	# A.organizational performance
	AGENT.p_org = Calculating_Performance_Org(AGENT,AGENT.knw_opt_raw, AGENT.knw_opt,agent_list,iter)
	p_org_df.append(AGENT.p_org)

	# C. update network of agent
	AGENT.network = Update_Network(AGENT)

	# D. update agent information

	for agent in agent_list:
		agent.p_ind = agent.Calculate_Performance(agent.knw_dict, AGENT.knw_opt)
		agent.Update_Mu_Var()

	for agent in agent_list:
		agent.Calculate_Util()
		agent.Find_Neighbor()


	# E. calcuating rank based on agent information
	for agent in agent_list:
		agent.Calculate_Rank(agent_list)

	# F. Update Status
	for agent in agent_list:

		if agent.rank[0]/agent.rank[1] > 0.5: #under 50%
			agent.status = 1 #activating
		else:
			agent.status = 0 #sleep

	# B. organizational mean and variance
	mu_org_df.append(np.mean(AGENT.knw_org_raw))
	var_org_df.append(np.var(AGENT.knw_org_raw))
	avg_util_df.append(Calculate_Avg_Util(agent_list,100))


#%% (3) Show Result
# Organizational performance
print(p_org_df)
