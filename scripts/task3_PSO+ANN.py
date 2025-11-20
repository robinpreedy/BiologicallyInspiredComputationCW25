
import numpy as np
from task1_ANN import (
 load_concrete,
    init_params,
    forward,
    mae,
    pack_params,
    unpack_params,
    get_args)

# desired swam size usually 10 - 100 Line 1
swarmsize = 10

#Line 7
class Particle:
    def __init__(self):
        #initialize a particle with random velocities and a position that is an ANN weight and biases
        layers=[8, 16, 8, 1]
        activate=["relu", "tanh", "identity"]
        startVal= np.random.randint(0,100)
        steps=100

        # Arguments needed to run program
        args = get_args()
        # csv file that contains data
        csv_path = args.csv

        #Data sets that run the experiment
        X_train, Y_train, X_test, Y_test = load_concrete(csv_path)
        randomNo = np.random.default_rng(startVal)
        weight, biases = init_params(layers, randomNo)
        flat = pack_params(weight, biases)
        #stored for later unflattening
        W_shapes = [w.shape for w in weight]
        B_shapes = [b.shape for b in biases]

        w0, b0 = unpack_params(flat, W_shapes, B_shapes)
        
        self.weight = w0
        self.biases = b0
       
        pred0 = forward(X_train, self.weight, self.biases)
   
        self.position = self.weight, self.biases  #weight and biases which is a list of numbers that span all the layers
        self.velocity = np.random.uniform(0.0,1.0, 2)# Line 9 v^-> (velocity)
        self.flat = flat
        self.w_shape = W_shapes
        self.b_shape = B_shapes
        self.pred = pred0
        self.train = Y_train
        self.fPass = X_train
        self.generate = randomNo
       

def fitness_ANN(y_train,pred):
    return mae(y_train,pred )

def PSO (iter,swarmSize,fitnessFunc):
    #Boundaries
    boundary = [0.01, 5.0]
    #Tracker for the current iteration
    iterNum = 0
    # List of best positions  Line 10
    #stores a list of all best position found throughout all iterations
    best_positions = []
    # proportion of velocity to be retained Line 2
    # your inertia weight
    alpha = 0.7
    # proportion of personal best to be retained Line 3
    beta = 0.0
    # proportion of global best to be retained Line 5
    delta = 0.0
    # proportion of the informants' best to be retained Line 4
    gamma = 0.0
    # jump size of a particle   Line 6
    epsilon = 0.05
    #Cognitive variable (particle)
    c1 = 1.5
    #Social variable (swarm)
    c2 = 1.5
    # (informant)
    c3 = 1.5

    #Line 9 create a list of particles
    particles = [Particle() for _ in range(swarmSize)]
    #Line 12 Initialize the best positions and fitness values
    for particle in particles:
        #Line 14
        best_positions.append(unpack_params(particle.flat, particle.w_shape, particle.b_shape))
    #print(best_positions)
    #Line 13 Asses fitness of all particles
    best_fitness = np.array([fitnessFunc(particle.train,particle.pred) for particle in particles])
    
    #Line 15 Global best pos
    # switch to np.argmin for a minimising problem
    swarm_best_position = best_positions[np.argmax(best_fitness)]
    # switch to np.min for a minimising problem
    #Global best fitness
    swarm_best_fitness = np.max(best_fitness)
   

    # variable that stores the previous fittest position of the local particle
    # gets the best position of current particles by grabbing the particle that corresponds to the min or max fitness
    #beta needs to be an int or float, beta is the information that makes up the ANN that has the best accuracy determined by mae
    #beta = best_positions[np.argmax(best_fitness)]  # Line 17
    
    #Beta gamma and delta all use particle fitness instead of position as position is tuple and fitness is float
    beta = swarm_best_fitness
    #print(beta)

    # variable that stores the previous fittest position of informants of the Local particle position including the particle itself
    # Gamma might be the previous global fittest position
    gamma = swarm_best_fitness # Line 18

    # variable that stores the previous global fittest position
    delta = swarm_best_fitness # Line 19
    
    #Line 11 and Line 27 the iteration loop
    while iterNum < iter:

        for particle in particles:

            for i in range(2):  # Line 20
                # uses variables chosen for Congnitive, social and informant weights
                # generates a new velocity for the particle
                # Line 24 
                particle.velocity = alpha * particle.velocity + c1 * (beta - fitnessFunc(particle.train,particle.pred)) + c2 * (gamma - fitnessFunc(particle.train,particle.pred)) + c3 * (delta - fitnessFunc(particle.train,particle.pred))  # Line 24

        for particle in particles: #Line 25
            #updates the particles position
            #I want to change the weights and biases, doing this will result in a different displayed fitness/accuracy
            bestFlat = particle.flat
            change = bestFlat + (particle.generate.normal(0.05, 0.25, size=bestFlat.size))
            #print(fitnessFunc(particle.train,particle.pred))
            particle.weight, particle.biases = unpack_params(change, particle.w_shape, particle.b_shape) # Line 26
            particle.pred = forward(particle.fPass,particle.weight,particle.biases)
            #print(fitnessFunc(particle.train,particle.pred))

        # Evaluate fitness of each particle
        #print(particle.position)
        fitness_values = np.array([fitnessFunc(particle.train,particle.pred) for particle in particles])
        #print(fitness_values)
      
        # Update best positions and fitness values
        # change to < for minimzing quiestions or > for maximising
        improved_indices = np.where(fitness_values < best_fitness, 1, 0)
        indices = list(improved_indices)
        #print(indices)
        index = []
        count = 0
        for x in indices:
            if x ==1:
                index.append(count)
            count += 1
        #print(index)
        for i in range(len(index)):
            best_positions[index[i]] = particles[index[i]].position
            best_fitness[index[i]] = fitness_values[index[i]]
        if np.max(fitness_values) > swarm_best_fitness:
            swarm_best_position = particles[np.argmax(fitness_values)].position
            swarm_best_fitness = np.max(fitness_values)
        
        #update gamma to be previous global fittest position
        gamma = delta
        #update delta to be new global fittest position
        delta = swarm_best_fitness
        iterNum += 1

# particle that returns the global best fitness and global best position at the end 
    
    return delta, swarm_best_position # Line 28

#p1 = Particle()
#f1 = fitness_ANN(p1.train, p1.pred)
#print(f1)
Pso1 = PSO(100,swarmsize,fitness_ANN)
print(Pso1)


