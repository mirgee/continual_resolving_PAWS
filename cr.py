from itertools import product
import networkx as nx
import pandas as pd
import copy
import operator
import os
import random
import scipy as sp
from scipy import sparse
import numpy as np

max_depth = 2
T = 3
grid_dim_x = 26
grid_dim_y = 26
base = 0.0
route_length = 9000


def init():
	df = pd.read_csv("data/paws_mdp_out.txt", sep=" ")
	global dist
	dist = pd.read_csv("data/dist.gop", sep=" ", header=None)
	global graph
	graph = \
		nx.from_pandas_dataframe(df, source='node_from', target='node_to',
		                         edge_attr=['distance', 'animal_density', 'grid_cell_x', 'grid_cell_y'])
	graph = nx.subgraph(graph, nx.node_connected_component(graph, base))
	global sigma1
	sigma1 = [[1/(grid_dim_x*grid_dim_y)] * grid_dim_y] * grid_dim_x
	global avg_strat1
	avg_strat1 = [[0] * grid_dim_y] * grid_dim_x
	global regret1
	regret1 = [[0] * grid_dim_y] * grid_dim_x
	global route
	route = [] # Will store the best route

def cfr_player1():
	p1 = 1
	p2 = 1
	global avg_strat1
	for _ in range(T):
		cf_values1 = [[0] * grid_dim_y] * grid_dim_x
		accum_val = 0

		for (grid_x, grid_y) in product(range(grid_dim_x), range(grid_dim_y)):
			cf_values1[grid_x][grid_y] = \
				cfr_player2([base], grid_x, grid_y, sigma1[grid_x][grid_y] * p1, p2, route_length)
			accum_val += sigma1[grid_x][grid_y] * cf_values1[grid_x][grid_y]

		for (grid_x, grid_y) in product(range(grid_dim_x), range(grid_dim_y)):
			regret1[grid_x][grid_y] += p2 * (cf_values1[grid_x][grid_y] - accum_val)
			avg_strat1[grid_x][grid_y] += p1 * sigma1[grid_x][grid_y]

		regret_matching1()

	avg_strat1 = [avg_strat1[i][j]/T for (i, j) in product(range(grid_dim_x), range(grid_dim_y))]

	return avg_strat1


def cfr_player2(node_history, grid_x, grid_y, p1, p2, rem_dist):
	global route
	curr_node = node_history[-1]
	edges = graph.edges(curr_node)
	edges = [edge for edge in edges if (edge[0], edge[1]) not in route or (edge[1], edge[0]) not in route]

	if rem_dist <= dist.iloc[int(curr_node), int(base)] or (len(node_history) > 1 and int(curr_node) == int(base)) \
			or len(edges) == 0:
		return compute_value_from_route(node_history, grid_x, grid_y)

	sigma2, subtree_nodes = get_empty_dict2(curr_node)
	regret2 = copy.deepcopy(sigma2)
	avg_strat2 = {edge: 0 for edge in edges}
	vals = copy.deepcopy(sigma2)

	for _ in range(T):
		vals = values(node_history, sigma2, vals, grid_x, grid_y, p1, p2, 0, rem_dist, [])
		# Update strategy for the whole subtree!
		regret2, sigma2, avg_strat2 = regret_matching2(sigma2, vals, curr_node, regret2, avg_strat2, subtree_nodes)

	next_edge = max(avg_strat2.items(), key=operator.itemgetter(1))[0]

	route.append(next_edge)

	return cfr_player2(node_history + [next_edge[1]], grid_x, grid_y, p1, avg_strat2[next_edge] * p2, rem_dist)


def values(node_history, sigma2, vals, grid_x, grid_y, p1, p2, d, rem_dist, subtree_visited):
	curr_node = node_history[-1]
	edges = graph.edges(curr_node)
	edges = [edge for edge in edges if (edge[0], edge[1]) not in route+subtree_visited or (edge[1], edge[0]) not in route+subtree_visited]

	if rem_dist <= dist.iloc[int(curr_node), int(base)] or (len(node_history) > 1 and int(curr_node) == int(base)) \
			or len(edges) == 0:
		vals[(node_history[-2], node_history[-1])] = compute_value_from_route(node_history, grid_x, grid_y)
		return vals

	if d > max_depth:
		vals[(node_history[-2], node_history[-1])] = heuristic(grid_x, grid_y, rem_dist)
		return vals

	for edge_index, edge in enumerate(edges):
		edge_data = graph[edge[0]][edge[1]]
		vals = values(node_history + [edge[1]], sigma2, vals, grid_x, grid_y, p1,
						sigma2[edge] * p2, d+1, rem_dist - edge_data['distance'], subtree_visited+[edge])

	return vals


def compute_value_from_route(route, grid_x, grid_y):
	value = 0
	for index, node in enumerate(route[:-1]):
		if route[index] != [route[index + 1]]:
			edge_data = graph[route[index]][route[index+1]]
			if edge_data['grid_cell_x'] == grid_x and edge_data['grid_cell_y'] == grid_y:
				value += edge_data['animal_density']
			else:
				value -= edge_data['animal_density']
	return value


def regret_matching1():
	regret = [[0] * grid_dim_y] * grid_dim_x
	for grid_x in range(grid_dim_x):
		for grid_y in range(grid_dim_y):
			regret[grid_x][grid_y] = max(regret1[grid_x][grid_y], 0)
	den = sum(map(sum, regret))
	if den > 0:
		for grid_x in range(grid_dim_x):
			for grid_y in range(grid_dim_y):
				sigma1[grid_x][grid_y] = regret[grid_x][grid_y] / den
	else:
		for grid_x in range(grid_dim_x):
			for grid_y in range(grid_dim_y):
				sigma1[grid_x][grid_y] = 1/(grid_dim_x*grid_dim_y)


def regret_matching2(sigma2, vals, curr_node, regret2, avg_strat2, subtree_nodes):
	# regret = {key: 0 for key in regret2.keys()}
	regret = {}
	for edge, _ in regret2.items():
		regret[edge] = max(regret2[edge], 0)
	den = sum(regret.values())

	node_vals = {}
	for node in subtree_nodes:
		node_vals[node] = 0
		for edge, value in vals.items():
			if node in edge:
				node_vals[node] += value/2


	for edge in regret2.keys():
		# We need counterfactual values per node, not edges! TODO: Sum up CF values of the edges.
		regret2[edge] += node_vals[edge[1]] - node_vals[edge[0]]
		sigma2[edge] = regret[edge] / den if den > 0 else 1/len(regret2)
		if edge in graph.edges(curr_node):
			avg_strat2[edge] += sigma2[edge] / T
	return regret2, sigma2, avg_strat2



def route_from_edges(edges):
	i = 0
	ret = []
	for edge in edges:
		ret.append(edge[i])
		i = ~i
	return ret


def get_empty_dict2(curr_node):
	visited_nodes = [curr_node]
	oriented = graph.to_directed()
	# edges = [edge for edge in oriented.edges() if (edge[0], edge[1]) not in route or (edge[1], edge[0]) not in route]
	edges = graph.edges(curr_node)
	to_visit = graph.neighbors(curr_node)
	for d in range(max_depth-1):
		while len(to_visit) > 0:
			node = to_visit.pop()
			visited_nodes += [node]
			to_visit += [n for n in graph.neighbors(node) if (n not in visited_nodes and n not in to_visit)]
			edges += [edge for edge in oriented.edges(node) if (((edge[0], edge[1]) not in route or (edge[1], edge[0]) not in route)
						and edge not in edges)]
	return {edge: 0 for edge in edges}, visited_nodes

def heuristic(grid_x, grid_y, rem_dist):
	cmd = 'mpirun -n 2 python2.7 ./loader.py op dist.gop 1 100 logfile ' + str(rem_dist)
	os.system(cmd)
	with open('logfile') as f:
		ret = f.readline().split('\t')
	return float(ret[2])

init()
print(cfr_player1())
print(route)