
import numpy as np

# desired swam size usually 10 - 100 Line 1
swarmsize = 100
# Dimension the size of plane that the swarm can move on and where the particles get their position from
dimension = 2 # works with 1 and 2 dimensions, when runing 1 dimensions n = len(x) error may occur

#Line 7
class Particle:
    def __init__(self, dim ):
        #initialize a particle with random weights and velocities
        self.position = np.random.uniform(0.0,1.0, dim) #Line 9  x^-> (position)
        self.velocity = np.random.uniform(0.0,1.0, dim)# Line 9 v^-> (velocity)

def fitness_Rastrign(x):
    #print(x)
    n = len(x)
    return 10 * n + sum([xi ** 2 - 10 * np.cos(2 * np.pi * xi) for xi in x])

def PSO (iter,swarmSize,fitnessFunc, dim):
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
    epsilon = 1


    #Line 9 create a list of particles
    particles = [Particle(dim) for _ in range(swarmSize)]

    #Line 12 Initialize the best positions and fitness values
    for particle in particles:
        #Line 14
        best_positions.append(np.copy(particle.position))

    #Line 13 Asses fitness of all particles
    best_fitness = np.array([fitnessFunc(particle.position) for particle in particles])
    
    #Line 15 Global best pos
    # switch to np.argmin for a minimising problem
    swarm_best_position = best_positions[np.argmin(best_fitness)]
    # switch to np.min for a minimising problem
    #Global best fitness
    swarm_best_fitness = np.min(best_fitness)


    # variable that stores the previous fittest position of the local particle
    # gets the best position of current particles by grabbing the particle that corresponds to the min or max fitness
    beta = best_positions[np.argmin(best_fitness)]  # Line 17

    # variable that stores the previous fittest position of informants of the Local particle position including the particle itself
    # Gamma might be the previous global fittest position

    gamma = swarm_best_position # Line 18

    # variable that stores the previous global fittest position
    delta = swarm_best_position # Line 19

    #Line 11 and Line 27 the iteration loop
    while iterNum < iter:

        for particle in particles:

            for i in range(dim):  # Line 20
                # Random number from 0.0 to beta inclusive
                # your cognitive weight
                b = np.random.uniform(0.0, beta, dim)  # Line 21
                # Random number from 0.0 to gamma inclusive (the largest position of the informants of the best position)
                c = np.random.uniform(0.0, gamma, dim)  #  Line 22
                #your social weight component
                # Random number from 0.0 to delta inclusive
                d = np.random.uniform(0.0, delta, dim)# Line 23
                # generates a new velocity for the particle
                # Line 24 
                particle.velocity = alpha * particle.velocity + b * (beta - particle.position) + c * (gamma - particle.position) + d * (delta - particle.position)  # Line 24

        for particle in particles: #Line 25
            particle.position = particle.position + (epsilon * particle.velocity)  # Line 26
            # basic boundary handling
            for i in range(len(particle.position)):
                if (particle.position[i] < boundary[0]):    # Min Boundary
                    particle.position[i] =  boundary[0]
                if (particle.position[i] > boundary[1]):    # Max Boundary
                    particle.position[i] =  boundary[1]

        # Evaluate fitness of each particle
        #print(particle.position)
        fitness_values = np.array([fitnessFunc(particle.position) for particle in particles])


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
        if np.min(fitness_values) < swarm_best_fitness:
            swarm_best_position = particles[np.argmin(fitness_values)].position
            swarm_best_fitness = np.min(fitness_values)

        #update gamma to be previous global fittest position
        gamma = delta
        #update delta to be new global fittest position
        delta = swarm_best_position.tolist()
        iterNum += 1

# particle that returns the global best fitness and global best position at the end 
    return delta # Line 28


p1 = Particle(dimension)
f1 = fitness_Rastrign(p1.position)
Pso1 = PSO(120,swarmsize,fitness_Rastrign, dimension)
print(Pso1)

