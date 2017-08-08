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
		print(rem_dist)
		feasible = [j for j in unvisited if dist.iloc[curr_node,j] <= rem_dist and dist.iloc[curr_node,j] != 0]
		A = [(scores.iloc[j]['S']/dist.iloc[curr_node,j])**4 for j in feasible]
		den = sum(A)
		if den == 0:
			continue
		P = [A[j]/den for j in range(len(A))]
		# P = sorted(P)
		# y = random.uniform(0,1)
		# next_node = next(i for i,v in enumerate(P) if v > y)
		next_node = np.random.choice(feasible, 1, p=P)[0]
		unvisited -= {next_node}
		rem_dist -= dist.iloc[curr_node, next_node]
		total_score += scores.iloc[curr_node, 0]
		curr_node = int(next_node)

	return total_score + scores.iloc[end_node, 0]

def tsiligirides(dist, scores):
	start_node = 0
	end_node = 1
	tmax = 9000
	max_res = 0
	for _ in range(3):
		res = generate_route(dist, scores, start_node, end_node, tmax)
		if res > max_res:
			max_res = res
	print(max_res)
	return max_res


dist = pd.read_csv("dist.gop", delim_whitespace=True, quoting=csv.QUOTE_NONE, index_col=None, header=0, engine='python', encoding = 'utf-8')
scores = pd.read_csv("scores.csv")
start = time.time()
tsiligirides(dist, scores)
end = time.time()
print(end - start)
