import pandas as pd
import random

def generate_route(dist, scores, start_node, end_node, tmax):
	curr_node = start_node
	rem_dist = tmax
	unvisited = set(range(len(scores))) - set(end_node)
	total_score = 0
	while rem_dist < dist[curr_node][end_node]:
		A = [(scores(j)/dist[curr_node,j])**4 for j in unvisited if dist[curr_node,j] <= rem_dist]
		den = sum(A)
		P = [A(j)/den for j in range(A)]
		P = sorted(P)
		y = random.uniform(0,1)
		next_node = next(i for i,v in enumerate(P) if v > y)
		unvisited -= set(next_node)
		rem_dist -= dist[curr_node][next_node]
		total_score += scores[curr_node]
		curr_node = next_node
	return total_score + scores[end_node]



dist = pd.from_csv("dist.gop", separator=" ")
scores = []

start_node = 0
end_node = 300
tmax = 500
max_res = 0
for _ in range(3000):
	res = generate_route(dist, scores, start_node, end_node, tmax)
	if res > max_res:
		max_res = res