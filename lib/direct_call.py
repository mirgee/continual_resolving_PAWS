import pyximport; pyximport.install()
import op
import fileformat
import grasp
import trajectory
import relink
import monitor
import sys
from mpi4py import MPI

comm = MPI.COMM_WORLD

fr = fileformat.GOPReader('dist.gop')
iters = 100
dlim = 9000
matrix = fr.get_distmatrix()

problem = op.OPProblem(
        [ op.OPItem(i, x[0], 0.0, matrix[i])
            for i, x in enumerate(fr.get_scores()) ],
        fr.get_start(),
        fr.get_end(),
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