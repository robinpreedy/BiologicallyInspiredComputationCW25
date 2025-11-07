
import numpy as np

# desired swam size usually 10 - 100 Line 1
swarmsize = 10
# might be a variable that allows the algorithm to simulate each particle moving in 3d space, which would require a change in position allowing for multiple
# Dimension the size of plane that the swarm can move on and where the particles get their position from
dimension = 1

#Try to add boundary handling preferably bouncing particles off of the boundary created
#Parameters that a particle needs for the Pseudocode of algorithm 39 Line 7
# look into how to initialise the position and velocity of a particle
#'''
class Particle:
    def __init__(self, dim ):
        #initialize a particle with random weights and velocities
        self.position = np.random.uniform(-1,1, dim) #Line 9  x^-> (position)
        self.velocity = np.random.uniform(-1,1, dim)# Line 9 v^-> (velocity)
#'''
# Fitness function chosen
# check fitness rastrign alorithm !!!!!!!!
def fitness_Rastrign(x):
    fitnessval = 0.0
    n = len(x)
    return 10 * n + sum([xi ** 2 - 10 * np.cos(2 * np.pi * xi) for xi in x])
    #return fitnessval

def PSO (iter,swarmSize,fitnessFunc, dim):

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


    #particles = np.random.uniform(-5.12, 5.12, (swarmSize, dim))
    particles = [Particle(dim) for _ in range(swarmSize)]

    # Initialize the best positions and fitness values
    #best_positions = np.copy(particles)

    for particle in particles:
        best_positions.append(np.copy(particle.position))

    best_fitness = np.array([fitnessFunc(particle.position) for particle in particles])
    #Global best pos
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
    delta = swarm_best_position

    while iterNum < iter:

        for particle in particles:

            for i in range(dim):  # Line 20
                # Random number from 0.0 to beta inclusive
                # your cognitive weight
               # b = np.random.uniform(0.0, (beta[i].position, dimension))  # Line 21
                b = np.random.uniform(0.0, beta, dim)  # Line 21
                # Random number from 0.0 to gamma inclusive (the largest position of the informants of the best position)
                #  Line 22
                c = np.random.uniform(0.0, gamma, dim)
                #your social weight component
                # Random number from 0.0 to delta inclusive
                d = np.random.uniform(0.0, delta, dim)# Line 23
                # generates a new velocity for the particle
                # Pseudocode Line 24  new velocity = inertia weight * particle's current velocity + cognitive weight * (particles best position so far - particle position)
                particle.velocity = alpha * particle.velocity + b * (beta - particle.position) + c * (gamma - particle.position) + d * (delta - particle.position)  # Line 24
                #velocities = w * velocities + c1 * r1 * (best_positions - particles) + c2 * r2 * (swarm_best_position - particles)

        for particle in particles:
            particle.position = particle.position + (epsilon * particle.velocity)  # Line 26

        # Evaluate fitness of each particle
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
        delta = swarm_best_position
        iterNum += 1

# particle that returns the best fitness at the end # might be delta the globalbest
    return swarm_best_position, swarm_best_fitness

#--------------------------------------- Add boundary handling

p1 = Particle(dimension)
f1 = fitness_Rastrign(p1.position)
Pso1 = PSO(100,10,fitness_Rastrign, dimension)
print(Pso1)
