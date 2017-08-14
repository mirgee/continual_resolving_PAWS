import networkx as nx
import numpy as np
import pandas as pd
"""
How do we represent strategy? Maybe as a vector of distributions for each edge?
But there is no need for that! We need strategy at each resolve step!
But both strategy and regrets are passed down the tree!

CF values are assigned to (node,edge) combination for defender, for grid cell attack for attacker.
For opponent, value is fixed over grid cells, his range should be updated after finding a trap?

How about ranges? Opponent: which cell he attacked. Defender: How to represent distribution over past routes?
Range is used to compute values in leaf nodes and is updated by strategy -> again, vector of distributions for each edge
-> equivalent to strategy?


"""

max_depth = 4
T = 10

def load_data():
	df = pd.read_csv("/home/miroslav/Source/research_task/paws_mdp_out.txt")
	dist = pd.read_csv("/home/miroslav/Source/research_task/dist.gop")
	# Grid should contain:
	#  reward, penalty for covering, not covering for def., attacker
	#  CFVs for attacker
	#  % covered by defender
	#  booleans for attacked (att. inform.), defended, attack prevented
	grid = [[0] * grid_dim_y] * grid_dim_x
	graph = nx.from_pandas_dataframe(df, edge_attr=['distance', 'animal_density', '	grid_cell_x','grid_cell_y'])
	return graph, dist, grid


def find_strategy(graph, n=5):
	""""Returns (a list of n tuples of probability) just one route from 0 to 0 for now"""
	curr_node = 0
	route = []
	while rem_distance > distance_to_base:
		distance_to_base = dist[0][curr_node]
		# edge, next_node, r1, v2 = resolve(curr_node, r1, v2, I)
		curr_node, edge, graph = resolve(curr_node, graph)
		route.append(edge)
		rem_distance -= edge.distance
	return route


def resolve(curr_node, graph):
	"""r1: route history, v2: opponent's cf values he would get by following computed strategy"""
	# Init random strategy and r2
	edges = graph.edges(curr_node)
	# grid = [[(1/(grid_dim_x*grid_dim_y), 1/(grid_dim_x*grid_dim_y))] * grid_dim_y] * grid_dim_x
	# We will average strategy - need all of them
	strategy = [[1/len(edges)] * len(edges)] * T
	regret = [0] * len(edges)
	# opp_regret = [0] * len(edges)
	# opp_range = [0] * len(edges)

	for t in range(T):
		graph, grid = values(curr_node, strategy, graph, grid, 0)
		graph, grid = update_subtree_strategies(curr_node, graph, grid, regret, 0)
		# Opp. regret?
	# Average strategies, sample (several) action(s), update r1, v2

def values(node, graph, grid, regret, d):
	"""d: depth"""
	if d == max_depth:
		return heuristic(r1, r2)

	edges = graph.edges(node)
	for edge in edges:
		# Update r1, r2 based on sigma

		v1, v2 = values(sigma, r1, r2, d+1)
		# update v1, v2, it depends who's time to act it is
	return v1, v2

def update_subtree_strategies():
	"""Compute regret and strategy"""
	pass