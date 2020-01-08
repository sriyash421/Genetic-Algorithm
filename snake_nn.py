from geneticNN import *
from game import *


def fit_func(policy):
    return game(policy, 100)


solver = GeneticAlgortithmNN(population_size=2000, num_layers=4, num_nodes=[
                             5, 125, 4],mutRate=0.01,selRate = 0.1, fitness_function=fit_func, filename="train.txt")

for i in range(1000):

#    print("\n Generation {} : \n".format(i))
    fitness_list = solver.calc_fitness()

    # print(fitness_list)

    parents = solver.select_mating_pool(fitness_list)

    print("\n Best Fitness : {}\n".format(solver.best_fitness))
#    time.sleep(1)
    offsprings = solver.crossover(parents)
    offsprings = solver.mutation(offsprings)
    solver.update_population(offsprings)
