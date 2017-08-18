from itertools import product
import networkx as nx
import pandas as pd
import copy
import operator

max_depth = 4
T = 20
grid_dim_x = 1
grid_dim_y = 2
base = 0.0
route_length = 5


def init():
	df = pd.read_csv("simple/nodes_list.txt", sep=" ")
	global dist
	dist = pd.read_csv("simple/dist_simple.gop", sep=" ")
	global graph
	graph = \
		nx.from_pandas_dataframe(df, source='node_from', target='node_to', edge_attr=['distance', 'animal_density',
														'grid_cell_x', 'grid_cell_y', 'sigma', 'regret', 'avg_strat'])
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
			cf_values1[grid_x][grid_y],_ = \
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

	sigma2 = get_empty_dict2(curr_node)
	regret2 = copy.deepcopy(sigma2)
	avg_strat2 = {edge: 0 for edge in edges}

	for _ in range(T):
		val = value(node_history, sigma2, grid_x, grid_y, p1, p2, 0, rem_dist)
		# Update strategy for the whole subtree!
		regrets, sigma2, avg_strat2 = regret_matching2(sigma2, val, curr_node, regret2, avg_strat2)

	next_edge = max(avg_strat2.iteritems(), key=operator.itemgetter(1))[0]

	route = route + [next_edge]

	return cfr_player2(node_history + [next_edge], grid_x, grid_y, p1, avg_strat1[next_edge] * p2, rem_dist)


def value(node_history, sigma2, grid_x, grid_y, p1, p2, d, rem_dist, subtree_visited):
	curr_node = node_history[-1]
	edges = graph.edges(curr_node)
	edges = [edge for edge in edges if (edge[0], edge[1]) not in route+subtree_visited or (edge[1], edge[0])
																						not in route+subtree_visited]

	if rem_dist <= dist.iloc[int(curr_node), int(base)] or (len(node_history) > 1 and int(curr_node) == int(base)) \
			or len(edges) == 0:
		return compute_value_from_route(node_history, grid_x, grid_y), node_history

	if d > max_depth:
		# Call heuristic
		pass

	for edge_index, edge in enumerate(edges):
		edge_data = graph[edge[0]][edge[1]]
		val = value(node_history + [edge[1]], grid_x, grid_y, p1,
						sigma2[edge] * p2, d+1, rem_dist - edge_data['distance'], subtree_visited+[edge])

	return val


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
				sigma1[grid_x][grid_y] = 1/(grid_x*grid_y)


def regret_matching2(sigma2, val, curr_node, regret2, avg_strat2):
	regret = {}
	for edge, regret in regret2:
		regret[edge].append(max(regret2[edge], 0))
	den = sum(regret.values())

	for edge in regret2.keys():
		regret2[edge] += val[edge[1]] - val[edge[0]]
		sigma2[edge] = regret[edge] / den if den > 0 else 1/len(regret2)
		if edge in graph.edges(curr_node):
			avg_strat2[edge] += sigma2 / T



def route_from_edges(edges):
	i = 0
	ret = []
	for edge in edges:
		ret.append(edge[i])
		i = ~i
	return ret


def get_empty_dict2(curr_node):
	visited_nodes = [curr_node]
	oriented = graph.to_oriented()
	edges = [edge for edge in oriented.edges() if (edge[0], edge[1]) not in route or (edge[1], edge[0]) not in route]
	for d in range(max_depth):
		for node in visited_nodes:
			visited_nodes += graph.neigbors(node)
			edges += [edge for edge in oriented.edges(node) if ((edge[0], edge[1]) not in route or (edge[1], edge[0]) not in route)
						and edge not in edges]
	return {edge: 0 for edge in edges}