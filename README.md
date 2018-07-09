# Continual re-solving and countrefactual regret minimization for PAWS

This is a project attempting to apply the technique for finding Nash equilibria called continual re-solving, successful on games like [poker](https://arxiv.org/abs/1701.01724), onto the domain of [Protection Assistant for Wildlife Security](https://www.cais.usc.edu/projects/wildlife-security/), a game modelling real world competition between rangers and poachers in a protected area.

Report on the first stage of the project can be found [here](http://docdro.id/rHXji7a).

This algorithm allows testing different metaheuristics for the problem. The most efficient one is the default: GRILS.

To run, first compile GRILS library:

In lib, first
`./pyxify.sh`
and then
`python2.7 setup.py build_ext --inplace`

and then:

`mpirun -n 2 python2.7 cr.py`.

The output is first the generated game graph, and then sequence of found traversed edges, found route, its distance, reward, and computation time.

`notebooks/` contains manipulation required to convert available data to a useful form. `cfr.py` 
