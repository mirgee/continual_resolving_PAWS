from itertools import product
import networkx as nx
import pandas as pd

max_depth = 4
T = 10
grid_dim_x = 1
grid_dim_y = 2
base = 0
route_length = 3


def init():
	df = pd.read_csv("/home/miroslav/Source/research_task/simple/nodes_list.txt", sep=" ")
	global dist
	dist = pd.read_csv("/home/miroslav/Source/research_task/simple/dist_simple.gop", sep=" ")
	global graph
	graph = \
		nx.from_pandas_dataframe(df, source='node_from', target='node_to', edge_attr=['distance', 'animal_density', 'grid_cell_x', 'grid_cell_y', 'sigma', 'regret', 'avg_strat'])
	global sigma1
	sigma1 = [[1/(grid_dim_x*grid_dim_y)] * grid_dim_y] * grid_dim_x
	global average_strat1
	average_strat1 = [[0] * grid_dim_y] * grid_dim_x
	global regret1
	regret1 = [[0] * grid_dim_y] * grid_dim_x


def cfr_player1():
	p1 = 1
	p2 = 1
	global average_strat1
	for _ in range(T):
		cf_values1 = [[0] * grid_dim_y] * grid_dim_x
		accum_val = 0

		for (grid_x, grid_y) in product(range(grid_dim_x), range(grid_dim_y)):
			cf_values1[grid_x][grid_y],_ = \
				cfr_player2([base], grid_x, grid_y, sigma1[grid_x][grid_y] * p1, p2,
				            route_length)
			accum_val += sigma1[grid_x][grid_y] * cf_values1[grid_x][grid_y]

		for (grid_x, grid_y) in product(range(grid_dim_x), range(grid_dim_y)):
			regret1[grid_x][grid_y] += p2 * (cf_values1[grid_x][grid_y] - accum_val)
			average_strat1[grid_x][grid_y] += p1 * sigma1[grid_x][grid_y]

		regret_matching1()

	average_strat1 = [average_strat1[i][j]/T for (i, j) in product(range(grid_dim_x), range(grid_dim_y))]
	return average_strat1


def cfr_player2(node_history, grid_x, grid_y, p1, p2, rem_dist):
	curr_node = node_history[-1]

	if rem_dist <= dist.iloc[int(curr_node), int(base)] or (len(node_history) > 1 and int(curr_node) == int(base)):
		# route = route_from_edges(edge_history)
		return compute_value_from_route(node_history, grid_x, grid_y), node_history

	edges = graph.edges(curr_node)
	cf_values2 = [0] * len(edges)
	routes = [[]] * len(edges)
	cf_value_curr = 0
	for edge_index, edge in enumerate(edges):
		edge_data = graph[edge[0]][edge[1]]
		node_history.append(edge[1])
		value, route = cfr_player2(node_history, grid_x, grid_y, p1,
						edge_data['sigma'] * p2, rem_dist - edge_data['distance'])
		cf_values2[edge_index] = value
		# I = route_from_edges(edge_history) + route
		cf_value_curr += edge_data['sigma'] * value
		routes.append(route)

	for edge_index, edge in enumerate(edges):
		edge_data = graph[edge[0]][edge[1]]
		edge_data['regret'] += p1 * (cf_values2[edge_index] - cf_value_curr)
		edge_data['avg_strat'] += p2 * edge_data['sigma']

	regret_matching2(curr_node)

	return cf_value_curr, node_history


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


def regret_matching2(curr_node):
	edges = graph.edges(curr_node)
	regret = {}
	for edge in edges:
		edge_data = graph[edge[0]][edge[1]]
		regret[edge] = max(edge_data['regret'], 0)
	den = sum(regret.values())
	if den > 0:
		for edge_index, edge in enumerate(edges):
			edge_data = graph[edge[0]][edge[1]]
			edge_data['sigma'] = edge_data['regret'] / den
	else:
		for edge_index, edge in enumerate(edges):
			edge_data = graph[edge[0]][edge[1]]
			edge_data['sigma'] = 1/len(edges)


def route_from_edges(edges):
	i = 0
	ret = []
	for edge in edges:
		ret.append(edge[i])
		i = ~i
	return ret


init()
print(cfr_player1())
for n1, n2 in graph.edges_iter():
    print(n1, n2, graph[n1][n2]['avg_strat']/T)
