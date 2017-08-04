import pandas as pd
import random
import csv

def generate_route(dist, scores, start_node, end_node, tmax):
	curr_node = start_node
	rem_dist = tmax
	unvisited = set(range(len(scores))) - {end_node}
	total_score = 0
	while rem_dist < dist.iloc[curr_node, end_node]:
		print (dist.iloc[curr_node, end_node])
		A = [(scores.iloc[j]['S']/dist.iloc[curr_node,j])**4 for j in unvisited if dist.iloc[curr_node,j] <= rem_dist]
		den = sum(A)
		P = [A[j]/den for j in range(A)]
		P = sorted(P)
		y = random.uniform(0,1)
		next_node = next(i for i,v in enumerate(P) if v > y)
		unvisited -= set(next_node)
		rem_dist -= dist[curr_node][next_node]
		total_score += scores[curr_node]
		curr_node = next_node
	return total_score + scores.iloc[end_node]['S']



dist = pd.read_csv("dist.gop", delim_whitespace=True, quoting=csv.QUOTE_NONE, index_col=None, header=0, engine='python', encoding = 'utf-8')
scores = pd.read_csv("scores.csv")

start_node = 0
end_node = 521
tmax = 500
max_res = 0
for _ in range(3000):
	res = generate_route(dist, scores, start_node, end_node, tmax)
	if res > max_res:
		max_res = res
print(max_res)