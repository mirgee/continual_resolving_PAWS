from itertools import product
from random import sample
import os
import networkx as nx
import numpy as np
import pandas as pd

max_depth = 4
T = 10
grid_dim_x = 1
grid_dim_y = 2
route_length = 40
base = 0
num_routes = 4

def init():
	df = pd.read_csv("/home/miroslav/Source/research_task/simple/nodes_list.txt", sep=" ")
	global dist
	dist = pd.read_csv("/home/miroslav/Source/research_task/simple/dist_simple.gop", sep=" ")
	global graph
	graph = \
		nx.from_pandas_dataframe(df, source='node_from', target='node_to', edge_attr=['distance', 'animal_density', 'grid_cell_x', 'grid_cell_y'])
	global sigma1
	sigma1 = [[(1/(grid_dim_x*grid_dim_y), 1/(grid_dim_x*grid_dim_y))] * grid_dim_y] * grid_dim_x
	global average_strat1
	accum_strat1 = [[(1/(grid_dim_x*grid_dim_y), 1/(grid_dim_x*grid_dim_y))] * grid_dim_y] * grid_dim_x
	global regret1
	regret1 = [[0] * grid_dim_y] * grid_dim_x
	global sigma2
	sigma2 = {} # Key: repr(routes), value: probability
	global regret2
	regret2 = {} # Key: (routes), value: regret
	global average_strat2
	average_strat2 = {}
	return graph, dist

def cfr_player1():
	p1 = 1
	p2 = 1
	for _ in range(T):
		cf_values1 = [[0] * grid_dim_y] * grid_dim_x
		accum_val = 0

		for (grid_x, grid_y) in product(range(grid_dim_x), range(grid_dim_y)):
			cf_values1[grid_x][grid_y] = /
				cfr_player2([], grid_x, grid_y, sigma1[grid_x][grid_y] * p1, p2,
				            route_length)
			accum_val += sigma1[grid_x][grid_y] * cf_values1[grid_x][grid_y]

		for (grid_x, grid_y) in product(range(grid_dim_x), range(grid_dim_y)):
			regret1[grid_x][grid_y] += p2 * (cf_values1[grid_x][grid_y] - accum_val)
			average_strat1 += p1 * sigma1[grid_x][grid_y]

		regret_matching1()

	average_strat1 = [average_strat1[i]/T for i in range(len(average_strat1))]


def cfr_player2(edge_history, grid_x, grid_y, p1, p2, rem_dist):
	curr_node = edge_history[-1][1]

	if rem_dist <= dist.iloc[curr_node, base] or curr_node == base:
		route = route_from_edges(edge_history)
		return compute_value_from_route(edge_history, grid_x, grid_y), route

	edges = graph.edges(curr_node)
	cf_values2 = [0] * len(edges)
	regret2 = [0] * len(edges)
	routes = [] * len(edges)
	cf_value_curr = 0
	for edge_index, edge in enumerate(edges):
		edge_data = graph[edge[0]][edge[1]]
		value, route = cfr_player2(edge_history.append(edge), grid_x, grid_y, p1,
						edge_data['sigma'] * p2, rem_dist - edge_data['distance'])
		cf_values2[edge_index] = value
		# I = route_from_edges(edge_history) + route
		cf_value_curr += sigma2[repr(route)] * value
		routes.append(route)

	for edge_index, edge in enumerate(edges):
		regret2[edge_index] += p1 * (cf_values2[edge_index] - cf_value_curr)
		average_strat2[repr(routes[edge_index])] += p2 * sigma2[repr(routes[edge_index])]

	regret_matching2(curr_node)

	return cf_value_curr, edge_history


def compute_value_from_route(route, grid_x, grid_y):
	value = 0
	for edge in route:
		edge_data = graph[edge[0]][edge[1]]
		if edge_data['grid_x'] == grid_x and edge_data['grid_y'] == grid_y:
			value += edge_data['animal_density']
		else:
			value -= edge_data['animal_density']
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


def route_from_edges(edges):
	i = 0
	ret = []
	for edge in edges:
		ret.append(edge[i])
		i = ~i
	return ret


init()
print(average_strat1)