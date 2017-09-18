import pyximport; pyximport.install()
from itertools import product
import networkx as nx
import pandas as pd
import copy
import operator
import os
import random
import numpy as np
import matplotlib.pyplot as pp
from collections import deque
import time
from lib import op, monitor
from mpi4py import MPI


def init_random(edges_per_node): # max_depth, T, route_length, num_nodes, edges_per_node):
	global max_depth
	global T
	global grid_dim_x
	global grid_dim_y
	global base
	global route_length

	# max_depth = 2
	# T = 3
	grid_dim_x = 5
	grid_dim_y = 5
	base = 0
	# route_length = 40

	global graph
	global untraversed_graph
	global dist
	global sigma1
	global avg_strat1
	global regret1
	global route
	global num_nodes
	global total_distance
	global total_reward
	global items

	# num_nodes = 20
	total_distance = 0
	total_reward = 0

	graph = nx.powerlaw_cluster_graph(num_nodes, edges_per_node, 0.3)
	graph = nx.subgraph(graph, nx.node_connected_component(graph, base))
	for (u, v) in graph.edges():
		graph.edge[u][v]['distance'] = 0 if u == v else random.randint(1, 10)
		graph.edge[u][v]['animal_density'] = random.randint(0, 20)
		graph.edge[u][v]['grid_cell_x'] = random.randint(0, grid_dim_x-1)
		graph.edge[u][v]['grid_cell_y'] = random.randint(0, grid_dim_y-1)
	# dist = nx.floyd_warshall_numpy(graph, weight='distance').tolist()
	# scores = []
	# with open('dist_rand.gop', 'w') as f:
	# 	f.write(str(num_nodes) + " 1 " + str(int(base)) + " " + str(int(base)) + "\n")
	# with open('dist_rand.gop', 'ab') as f:
	# 	for line in np.matrix(dist):
	# 		np.savetxt(f, line, fmt='%.2f')
	# with open('dist_rand.gop', 'a') as f:
	# 	for _ in range(num_nodes):
	# 		r = random.randint(0, 10)
	# 		f.write(str(r) + '\n')
	# 		scores.append(r)

	graph, dist, scores = create_extended_graph(graph)

	untraversed_graph = graph.copy()

	nx.draw_networkx(graph, pos=nx.spring_layout(graph))
	pp.show()

	sigma1 = [[1 / (grid_dim_x * grid_dim_y)] * grid_dim_y] * grid_dim_x
	avg_strat1 = [[0] * grid_dim_y] * grid_dim_x
	regret1 = [[0] * grid_dim_y] * grid_dim_x
	route = []
	items = [op.OPItem(i, x, 0.0, dist[i]) for i, x in enumerate(scores)]


def init_simple():
	global max_depth
	global T
	global grid_dim_x
	global grid_dim_y
	global base
	global route_length
	global total_distance
	global total_reward

	total_distance = 0
	total_reward = 0
	T = 4 # Num. of iterations
	grid_dim_x = 1
	grid_dim_y = 2
	base = 0.0 # Starting node number
	route_length = 10 # Distance limit

	global graph
	global dist
	global sigma1
	global avg_strat1
	global regret1
	global route

	df = pd.read_csv("simple/nodes_list.txt", sep=" ")
	dist = pd.read_csv("simple/dist_simple.gop", sep=" ", header=None)
	graph = \
		nx.from_pandas_dataframe(df, source='node_from', target='node_to',
		                         edge_attr=['distance', 'animal_density', 'grid_cell_x', 'grid_cell_y'])
	sigma1 = [[1 / (grid_dim_x * grid_dim_y)] * grid_dim_y] * grid_dim_x
	avg_strat1 = [[0] * grid_dim_y] * grid_dim_x
	regret1 = [[0] * grid_dim_y] * grid_dim_x
	route = []


def init():
	global max_depth
	global T
	global grid_dim_x
	global grid_dim_y
	global base
	global route_length
	global total_distance
	global total_reward

	total_distance = 0
	total_reward = 0
	max_depth = 2
	T = 3
	grid_dim_x = 26
	grid_dim_y = 26
	base = 0.0
	route_length = 9000

	global graph
	global dist
	global sigma1
	global avg_strat1
	global regret1
	global route

	df = pd.read_csv("data/paws_mdp_out.txt", sep=" ")
	dist = pd.read_csv("data/dist.gop", sep=" ", header=None)
	graph = \
		nx.from_pandas_dataframe(df, source='node_from', target='node_to',
		                         edge_attr=['distance', 'animal_density', 'grid_cell_x', 'grid_cell_y'])
	graph = nx.subgraph(graph, nx.node_connected_component(graph, base))
	sigma1 = [[1/(grid_dim_x*grid_dim_y)] * grid_dim_y] * grid_dim_x
	avg_strat1 = [[0] * grid_dim_y] * grid_dim_x
	regret1 = [[0] * grid_dim_y] * grid_dim_x
	route = []

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
	global total_distance
	global total_reward

	curr_node = node_history[-1]
	if len(node_history) > 2:
		untraversed_graph.remove_edge(node_history[-2], node_history[-1])

	edges = []
	for edge in untraversed_graph.edges(curr_node):
		next_node = edge[0] if edge[1] == curr_node else edge[1]
		edge_data = untraversed_graph[curr_node][next_node]
		untraversed_graph.remove_edge(curr_node, next_node)
		try:
			if nx.shortest_path_length(untraversed_graph, next_node, base, 'distance') < rem_dist - edge_data['distance']:
				edges.append(edge)
		except:
			pass
		untraversed_graph.add_edge(curr_node, next_node, attr_dict=edge_data)

	# if rem_dist <= nx.shortest_path_length(untraversed_graph, curr_node, base, 'distance')
	if len(node_history) > 1 and curr_node == base:
		return compute_value_from_route(node_history, grid_x, grid_y)

	if len(edges) == 0:
		nx.draw_networkx(graph, pos=nx.spring_layout(graph))
		pp.show()
		nx.draw_networkx(untraversed_graph, pos=nx.spring_layout(untraversed_graph))
		pp.show()
		raise Exception("THIS SHOULDN'T HAPPEN")

	sigma2, subtree_nodes = get_empty_dict2_eff(curr_node)
	regret2 = copy.deepcopy(sigma2)
	avg_strat2 = {edge: 0 for edge in edges}
	vals = copy.deepcopy(sigma2)


	for _ in range(T):
		vals = values(node_history, sigma2, vals, grid_x, grid_y, p1, p2, 0, rem_dist, [])
		regret2, sigma2, avg_strat2 = regret_matching2(sigma2, vals, regret2, avg_strat2, subtree_nodes)

	next_edge = max(avg_strat2.items(), key=operator.itemgetter(1))[0]

	print(next_edge)

	route.append(next_edge)
	total_distance += graph[next_edge[0]][next_edge[1]]['distance']
	# total_reward += graph[next_edge[0]][next_edge[1]]['animal_density']
	total_reward += graph.node[next_edge[1]]['animal_density']

	return cfr_player2(node_history + [next_edge[1]], grid_x, grid_y, p1, avg_strat2[next_edge] * p2,
	                   rem_dist - graph[next_edge[0]][next_edge[1]]['distance'])


def values(node_history, sigma2, vals, grid_x, grid_y, p1, p2, d, rem_dist, subtree_visited):
	curr_node = node_history[-1]

	local_untraversed_graph = untraversed_graph.copy()
	local_untraversed_graph.remove_edges_from(subtree_visited)

	edges = []
	for edge in local_untraversed_graph.edges(curr_node):
		next_node = edge[0] if edge[1] == curr_node else edge[1]
		edge_data = local_untraversed_graph[curr_node][next_node]
		local_untraversed_graph.remove_edge(curr_node, next_node)
		try:
			if nx.shortest_path_length(local_untraversed_graph, next_node, base, 'distance') < rem_dist - edge_data['distance']:
				edges.append(edge)
		except:
			pass
		local_untraversed_graph.add_edge(curr_node, next_node, attr_dict=edge_data)

	if len(node_history) > 1 and curr_node == base:
		vals[(node_history[-2], node_history[-1])] = compute_value_from_route(node_history, grid_x, grid_y)
		return vals

	if d > max_depth:
		vals[(node_history[-2], node_history[-1])] = heuristic(node_history[-1], grid_x, grid_y, rem_dist)
		return vals

	if len(edges) == 0:
		raise Exception("THIS SHOULDN'T HAPPEN")

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
			# If we wanted distr. over cells as input, just multiply here by prob. of attack
			if edge_data['grid_cell_x'] == grid_x and edge_data['grid_cell_y'] == grid_y:
				# value += edge_data['animal_density']
				value += graph.node[node]['animal_density']
			else:
				# value -= edge_data['animal_density']
				value -= graph.node[node]['animal_density']
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


def regret_matching2(sigma2, vals, regret2, avg_strat2, subtree_nodes):
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
		regret2[edge] += node_vals[edge[1]] - node_vals[edge[0]]
		sigma2[edge] = regret[edge] / den if den > 0 else 1/len(regret2)
		if edge in avg_strat2.keys():
			avg_strat2[edge] += sigma2[edge] / T
	return regret2, sigma2, avg_strat2



def route_from_edges(edges):
	i = 0
	ret = []
	for edge in edges:
		ret.append(edge[i])
		i = ~i
	return ret


def get_empty_dict2_slow(curr_node):
	visited_nodes = [curr_node]
	oriented = graph.to_directed()
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


def get_empty_dict2_eff(curr_node):
	node_queue = deque()
	node_queue.append(curr_node)
	visited_nodes = [curr_node]
	edges = []
	curr_depth = 0
	elems_to_depth_increase = 1
	next_elems_to_depth_increase = 0

	while len(node_queue) > 0:
		curr = node_queue.popleft()
		for edge in graph.edges(curr):
			if edge not in edges:
				edges.append(edge)
			if edge[1] not in visited_nodes:
				node_queue.append(edge[1])
				visited_nodes.append(edge[1])
		next_elems_to_depth_increase += len(graph.edges(curr))
		elems_to_depth_increase -= 1
		if elems_to_depth_increase == 0:
			curr_depth += 1
			if curr_depth > max_depth:
				break
			elems_to_depth_increase = next_elems_to_depth_increase
			next_elems_to_depth_increase = 0
	return {edge: 0 for edge in edges}, visited_nodes


def heuristic_old(curr_node, grid_x, grid_y, rem_dist):
	with open('dist_rand.gop', 'r') as f:
		lines = f.readlines()
		lines[0] = str(num_nodes) + " 1 " + str(int(curr_node)) + " " + str(int(base)) + "\n"
	with open('dist_rand.gop', 'w') as f:
		for line in lines:
			f.write(line)

	cmd = 'mpirun -n 2 python2.7 ./loader.py op dist_rand.gop 1 100 logfile ' + str(int(rem_dist))
	os.system(cmd)
	with open('logfile') as f:
		ret = f.readline().split('\t')
	try:
		return float(ret[2])
	except:
		return 0


def heuristic(curr_node, grid_x, grid_y, rem_dist):
	global items

	g = op.OP_GRASP_T(comm)

	problem = op.OPProblem(
		items,
		curr_node,
		0,
		rem_dist
	)

	solution = g.search(problem, 100)
	return int(solution.get_score())


def create_extended_graph(orig_graph):
	new_graph = nx.Graph()
	scores = []
	i = len(orig_graph.nodes())
	for edge in orig_graph.edges():
		new_graph.add_node(edge[0], animal_density=0, grid_cell_x=orig_graph[edge[0]][edge[1]]['grid_cell_x'],
		                   grid_cell_y=orig_graph[edge[0]][edge[1]]['grid_cell_y'])
		new_graph.add_node(edge[1], animal_density=0, grid_cell_x=orig_graph[edge[0]][edge[1]]['grid_cell_x'],
		                   grid_cell_y=orig_graph[edge[0]][edge[1]]['grid_cell_y'])
		new_graph.add_node(i, animal_density=orig_graph[edge[0]][edge[1]]['animal_density'],
		                   grid_cell_x=orig_graph[edge[0]][edge[1]]['grid_cell_x'],
		                   grid_cell_y=orig_graph[edge[0]][edge[1]]['grid_cell_y'])
		new_graph.add_edge(edge[0], i, distance=orig_graph[edge[0]][edge[1]]['distance'] / 2,
		                   grid_cell_x=orig_graph[edge[0]][edge[1]]['grid_cell_x'],
		                   grid_cell_y=orig_graph[edge[0]][edge[1]]['grid_cell_y'])
		new_graph.add_edge(i, edge[1], distance=orig_graph[edge[0]][edge[1]]['distance'] / 2,
		                   grid_cell_x=orig_graph[edge[0]][edge[1]]['grid_cell_x'],
		                   grid_cell_y=orig_graph[edge[0]][edge[1]]['grid_cell_y'])
		i += 1
	for node_num, node_data in new_graph.nodes(data=True):
		scores.append(node_data['animal_density'])

	dist = nx.floyd_warshall_numpy(new_graph, weight='distance').tolist()
	return new_graph, dist, scores


def test():
	global max_depth
	global T
	global route_length
	global num_nodes
	global route
	global total_distance
	global total_reward
	global untraversed_graph
	Ts = [2, 3, 4]
	max_depths = [2, 3, 4]
	route_lengths = [20, 30, 40, 50]
	node_counts = [20, 30, 40, 50]
	edges_per_node_list = [2, 3, 4]
	num_tests = 3

	with open('test_results', 'w') as f:
		f.write("T max_depth  route_length num_nodes edges_per_node\n")

	for num_nodes in node_counts:
		for edges_per_node in edges_per_node_list:
			init_random(edges_per_node)  # max_depth, T, route_length, node_count, num_nodes)
			for T in Ts:
				for max_depth in max_depths:
					for route_length in route_lengths:
						with open('test_results', 'a') as f:
							f.write("\n" + str(T) + " " + str(max_depth) + " " + str(route_length) + " " + str(num_nodes) + " " + str(edges_per_node) + "\n")
						for _ in range(num_tests):
							untraversed_graph = graph.copy()
							start = time.time()
							cfr_player2([base], 2, 4, 1, 1, route_length)
							# cfr_player1()
							end = time.time()
							with open('test_results', 'a') as f:
								f.write(str(end-start) + '\n')
								f.write(str(len(route)) + '\n')
								f.write(str(total_distance) + '\n')
							print(route)
							print(total_distance)
							print(total_reward)
							print(end-start)
							route = []
							total_distance = 0
							total_reward = 0
	with open('test_results', 'a') as f:
		f.write("Main dataset: \n")
	init()
	start = time.time()
	cfr_player2([base], 2, 4, 1, 1, 9000)
	end = time.time()
	with open('test_results', 'a') as f:
		f.write(str(end - start) + '\n')

comm = MPI.COMM_WORLD

if comm.Get_rank() == 0:
	logfile = open("logfile", "w")
	monitor.monitor_best(comm, logfile)
else:
	test()