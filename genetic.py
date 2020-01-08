import numpy as np
import multiprocessing

class GeneticAlgortithm() :

    def __init__(self, population_size, num_inputs, num_outputs, policy_range, fitness_function, offspring_size, parent_size, filename) :

        self.pop_size = population_size
        self.parent_size = parent_size
        self.offspring_size = offspring_size
        self.policy_size = (num_outputs, num_inputs)
        self.fitness_func = fitness_function
        self.range = policy_range
        self.population = np.random.uniform(low=-self.range, high=self.range, size =(population_size,self.policy_size[0],self.policy_size[1]))
        self.best_policy = np.empty(self.policy_size)
        self.best_fitness = 0
        self.generation = 0
        self.file = filename
        print("Class Initialised\n")
    

    def calc_fitness(self) :
        
        fitness_list = []
        p = multiprocessing.Pool(1)#multiprocessing.cpu_count())
        policy_list = []
        for i in range(self.pop_size) :
            policy = self.population[i][:][:]
            policy_list.append(policy)

        fitness = p.map(self.fitness_func, policy_list)

        p.close()
        p.join()

        for i in range(self.pop_size) :           
            fitness_list.append((fitness[i], policy_list[i]))
        return fitness_list

    def select_mating_pool(self, fitness_list) :
        self.generation+=1
        fitness_list = sorted(fitness_list, key = lambda x:x[0], reverse=True)
        parents = np.empty((self.parent_size,self.policy_size[0],self.policy_size[1]))

        for i in range(self.parent_size) :
            parents[i][:][:] = fitness_list[i][1][:][:]
        
        self.best_policy = fitness_list[0][1][:][:]
        self.best_fitness = fitness_list[0][0]
        self.write_to_file()
        return parents
    
    def write_to_file(self) :
        with open(self.file,"a") as output_file :
            output_file.write("\n Generation {} : \n Policy: {} \n Fitness: {}".format(self.generation, self.best_policy, self.best_fitness))

    def crossover(self, parents) :

        offsprings = np.empty((self.offspring_size,self.policy_size[0],self.policy_size[1]))

        for i in range(self.offspring_size) :
            id1 = i%parents.shape[0]
            id2 = (i+1)%parents.shape[0]
            weight_matrix = np.random.randn(self.policy_size)#np.random.choice([0,1],size=self.policy_size, p=[1/3.0,2/3.0])
            offspring = np.multiply(weight_matrix,parents[id1][:][:])+np.multiply(1-weight_matrix,parents[id2][:][:])

            offsprings[i][:][:] = offspring[:][:]
        
        return offsprings

    
    def mutation(self, offsprings) :

        for i in range(offsprings.shape[0]) :
            random_value = np.random.uniform(size=self.policy_size,high=self.range,low=-self.range)
            random_position = np.random.choice([0,1],p=[2/3.0,1/3.0],size=self.policy_size)

            offsprings[i][:][:] += np.multiply(random_position,random_value) 

        return offsprings
    
    def update_population(self, parents, offsprings) :

        self.population[:self.parent_size][:][:] = parents
        self.population[self.parent_size:][:][:] = offsprings[:self.pop_size-self.parent_size][:][:]