"""
First player to act is attacker, than only defenders act. This does not mean that attackers strategy must be given -
the cf values, regrets and optimal strategy will be calculated.

Attacker:
	CFV: expected values per grid cell
	Regrets: per grid cell
	Strategy: distribution over grid cells

Defender:
	CFV: per (node, edge) on graph, local vars in recursion
	Regrets: per (node, edge) on graph, local vars in recursion
	Strategy: distribution over edges per node, local var in recursion, used but used to compute global one
	Global strategy: distribution over edges per node, accumulated by regret matching

Grid: not explicit data structure, represented as matrices of:
	attacker, defender rewards
	attacker regrets
	% covered by defender

Graph: distance, animal density, grid_x, grid_y
"""
from itertools import product
from random import sample
import os
import networkx as nx
import numpy as np
import pandas as pd

max_depth = 4
T = 10
grid_dim_x = 15
grid_dim_y = 15
att_resources = 2
route_length = 9000
base = 0


def init():
	# TODO: Add edge attributes
	df = pd.read_csv("/home/miroslav/Source/research_task/paws_mdp_out.txt", sep=" ")
	global dist
	dist = pd.read_csv("/home/miroslav/Source/research_task/dist.gop", sep=" ")
	global graph
	graph = \
		nx.from_pandas_dataframe(df, source='node_from', target='node_to', edge_attr=['distance', 'animal_density', 'grid_cell_x', 'grid_cell_y', 'sigma', 'regret', 'accum_strat'])
	global sigma1
	sigma1 = [[(1/(grid_dim_x*grid_dim_y), 1/(grid_dim_x*grid_dim_y))] * grid_dim_y] * grid_dim_x
	global accum_strat1
	accum_strat1 = [[(1/(grid_dim_x*grid_dim_y), 1/(grid_dim_x*grid_dim_y))] * grid_dim_y] * grid_dim_x
	# global sigma2 = {} # (node, edge) -> probability, has to be initialized!!! HOW???
	# global accum_strat2 = {} # (node, edge) -> probability
	global regret1
	regret1 = [[0] * grid_dim_y] * grid_dim_x
	# global regret2 = {} # (node, edge) -> probability
	return graph, dist


def cfr_player1():
	p1 = 1
	p2 = 1
	for _ in range(T):
		cf_values1 = [[0] * grid_dim_y] * grid_dim_x
		accum_val = 0

		# for (grid_x, grid_y) in sample(product(range(len(grid_dim_x)), range(len(grid_dim_y))), att_resources):
		for (grid_x, grid_y) in product(range(grid_dim_x), range(grid_dim_y)):
			# -cfv1 = cfv2 ?
			route = cfr_player2(base, grid_x, grid_y, sigma1[grid_x][grid_y] * p1, p2, [], route_length)
			cf_values1[grid_x][grid_y] = -compute_value_from_route(route, grid_x, grid_y)
			accum_val += sigma1[grid_x][grid_y] * cf_values1[grid_x][grid_y]

		for (grid_x, grid_y) in product(range(grid_dim_x), range(grid_dim_y)):
			regret1[grid_x][grid_y] += p2 * (cf_values1[grid_x][grid_y] - accum_val)
			accum_strat1 += p1 * sigma1[grid_x][grid_y]
			regret_matching1()

	return route


def cfr_player2(curr_node, grid_x, grid_y, p1, p2, route, rem_dist):
	if rem_dist <= dist.iloc[curr_node, base]:
		return route

	edges = graph.edges(curr_node)
	cf_values2 = [0] * len(edges)

	# TODO: Here we should loop T times normalize accum. strategies & construct route from them! Or not?
	for edge_index, edge in enumerate(edges):
		# cf_values2[edge_index], route = cfr_player2(edge[1], grid_x, grid_y, p1, edge.sigma * p2, rem_dist - edge.distance, route, depth+1)
		edge_data = graph[edge[0]][edge[1]]
		cf_values2[edge_index] = \
			values(edge[1], grid_x, grid_y, p1, edge_data['sigma'] * p2, 0, rem_dist - edge_data['distance'])

	regret_matching2(curr_node)

	best_edge = argmax_attr(edges, 'sigma')
	route.append(best_edge)

	return cfr_player2(best_edge[1], grid_x, grid_y, p1, graph[best_edge[0]][best_edge[1]]['sigma'] * p2, route, rem_dist - dist.iloc[curr_node, best_edge[1]])


def values(curr_node, grid_x, grid_y, p1, p2, depth, rem_dist):
	if depth > max_depth or dist.iloc[curr_node, base] <= rem_dist:
		# TODO: Problem: we can't modify rem_dist without recompiling the program again.
		return heuristic(curr_node, grid_x, grid_y, rem_dist)

	accum_val = 0

	# TODO: Exclude parent node, or mark visited nodes?
	edges = graph.edges(curr_node)
	values = [0] * len(edges)
	for edge_index, edge in enumerate(edges):
		edge_data = graph[edge[0]][edge[1]]
		values[edge_index] = values(edge[1], grid_x, grid_y, edge_data['sigma'] * p2, depth+1, rem_dist - edge_data['distance'])

		accum_val += edge_data['sigma'] * values[edge_index]

	for edge_index, edge in enumerate(edges):
		edge_data['regret'] += p1 * (values[edge_index] - accum_val)
		# TODO: Do we need accum_strat? This is what approaches Nash! Or normalized sigmas?
		edge_data['accum_strat'] += p2 * edge_data['sigma']

	return accum_val

def compute_value_from_route(route, grid_x, grid_y):
	value = 0
	for edge in route:
		# TODO: Compute total a.d. per concerned grid cell
		total_animal_density = 1
		ratio = graph.edges(edge).animal_density / total_animal_density
		if edge.grid_x == grid_x and edge.grid_y == grid_y:
			value += ratio * values[edge.grid_x][edge.grid_y]
		else:
			value -= ratio * values[edge.grid_x][edge.grid_y]
	return value

def regret_matching1():
	den = sum(sum(regret1))
	if den > 0:
		for grid_x in regret1:
			for grid_y in grid_x:
				sigma1[grid_x][grid_y] = regret1[grid_x][grid_y] / den
	else:
		for grid_x in regret1:
			for grid_y in grid_x:
				sigma1[grid_x][grid_y] = 1/(grid_x*grid_y)


def regret_matching2(curr_node):
	edges = graph.edges(curr_node)
	den = sum(edges.regret)
	if den > 0:
		for edge in edges:
			edge.sigma = edge.regret / den
	else:
		for edge in edges:
			edge.sigma = 1/len(edges)

def argmax_attr(edges, attr):
	max_val = -9999
	max_index = (None, None)
	for edge in edges:
		val = graph[edge[0]][edge[1]][attr]
		if val > max_val:
			max_val = val
			max_index = edge
	return edge

def heuristic():
	os.system('mpirun -n 2 python2.7 ./loader.py op dist.gop 1 1 logfile')
	with open('logfile') as f:
		content = f.readlines()
	content = [x.strip() for x in content]
	return content[-1].split()[0]


init()
print(cfr_player1())
