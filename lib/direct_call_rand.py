import pyximport; pyximport.install()
import op
import fileformat
import grasp
import trajectory
import relink
import monitor
import sys
from mpi4py import MPI
import networkx as nx
import random
import numpy as np

comm = MPI.COMM_WORLD

num_nodes = 100
edges_per_node = 3
base = 0

graph = nx.powerlaw_cluster_graph(num_nodes, edges_per_node, 0.3)
graph = nx.subgraph(graph, nx.node_connected_component(graph, base))
for (u, v) in graph.edges():
	graph.edge[u][v]['distance'] = 0 if u == v else random.randint(1, 10)

dist = np.matrix(nx.floyd_warshall_numpy(graph, weight='distance')).tolist()

iters = 100
dlim = 9000
scores = []

for _ in range(num_nodes):
	scores.append(random.randint(0,10))

problem = op.OPProblem(
        [ op.OPItem(i, x, 0.0, dist[i])
            for i, x in enumerate(scores) ],
        0,
        0,
        0.0
    )

problem.set_capacity(dlim)

g = op.OP_GRASP_T(comm)

# g = op.OP_GRASP_I(comm)

if comm.Get_rank() == 0:
    # control process
    best = monitor.monitor_best(comm, sys.stdout)
else:
    solution = g.search(problem, iters)
    print(solution.get_score())