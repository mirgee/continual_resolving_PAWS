import pandas as pd
import numpy as np
import time
import csv

def generate_route(dist, scores, start_node, end_node, tmax):
	curr_node = start_node
	rem_dist = tmax
	unvisited = set(range(len(scores))) - {start_node, end_node}
	total_score = 0
	while rem_dist > dist.iloc[curr_node, end_node]:
		# print(rem_dist)
		feasible = np.fromiter((j for j in unvisited if (dist.iloc[curr_node,j] <= rem_dist and dist.iloc[curr_node,j] > 0 and dist.iloc[end_node, j] >= 0 and dist.iloc[end_node, j] <= (rem_dist - dist.iloc[curr_node, j]))), np.int)
		A = np.fromiter(((scores.iloc[j]['S']/dist.iloc[curr_node,j])**4 for j in feasible), np.float)
		den = np.sum(A)
		if den == 0:
			break
		P = np.divide(A, den)
		next_node = np.random.choice(feasible, 1, p=P)[0]
		total_score, unvisited = reward_from_path(curr_node, next_node, unvisited, total_score)
		unvisited -= {next_node}
		rem_dist -= dist.iloc[curr_node, next_node]
		total_score += scores.iloc[curr_node, 0]
		curr_node = int(next_node)

	total_score, unvisited = reward_from_path(curr_node, end_node, unvisited, total_score)
	return total_score

def reward_from_path(curr_node, next_node, unvisited, total_score):
	previous_node = next_node
	while previous_node != curr_node:
		previous_node = pred.iloc[curr_node][previous_node]
		if previous_node in unvisited:
			total_score += scores.iloc[previous_node, 0]
			unvisited -= {previous_node}
	return total_score, unvisited

def tsiligirides(dist, scores):
	start_node = 0
	end_node = 0
	tmax = 9000
	max_res = 0
	for _ in range(3000):
		res = generate_route(dist, scores, start_node, end_node, tmax)
		if res > max_res:
			max_res = res
			print(max_res, time.time() - start)
	print(max_res)
	return max_res


dist = pd.read_csv("data/dist.gop", delim_whitespace=True, quoting=csv.QUOTE_NONE, index_col=None, header=0, engine='python', encoding = 'utf-8')
pred = pd.read_csv("data/predecessors.txt", delim_whitespace=True, quoting=csv.QUOTE_NONE, index_col=None, header=0, engine='python', encoding = 'utf-8')
scores = pd.read_csv("scores.csv")
start = time.time()
tsiligirides(dist, scores)
end = time.time()
print(end - start)
