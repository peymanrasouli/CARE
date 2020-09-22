from deap.benchmarks.tools import convergence, diversity, hypervolume, igd


############### Benchmarks and Evaluations ####################

## Calculating Hypervolume
pops = logbook.select('pop')
pops_obj = [np.array([ind.fitness.wvalues for ind in pop]) * -1 for pop in pops]
ref = np.max([np.max(wobjs, axis=0) for wobjs in pops_obj], axis=0) + 1
hypervols = [hypervolume(pop, ref) for pop in pops]
plt.plot(hypervols)
plt.xlabel('Iterations')
plt.ylabel('Hypervolume')

## Calculating GD, IGD


## Calculating Convergence
