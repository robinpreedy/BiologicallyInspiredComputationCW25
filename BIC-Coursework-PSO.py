
import numpy as np

# desired swam size usually 10 - 100 Line 1
swarmsize = 10


#Try to add boundary handling preferably bouncing particles off of the boundary created
#Parameters that a particle needs for the Pseudocode of algorithm 39 Line 7
# look into how to initialise the position and velocity of a particle
class Particle:
    def __init__(self ):
    # initialize a particle with random weights and velocities
        self.position = round(np.random.uniform(0,1), 2) #Line 9  x^-> (position)
        self.velocity = round(np.random.uniform(0,1), 2) # Line 9 v^-> (velocity)

# Fitness function chosen
# check fitness rastrign alorithm !!!!!!!!
def fitness_Rastrign(xi):
    fitnessval = 0.0
   # for i in range(len(x)):
        #xi = x[i].position
    fitnessval += (xi * xi) - (10 * np.cos(2 * np.pi * xi)) + 10
    return fitnessval

def PSO (iter,swarmSize):

    #Tracker of current particle of the list of particles
    count = 0
    # best position variable Line 10
    Best = []
    # proportion of velocity to be retained Line 2
    # your inertia weight
    alpha = 1
    # proportion of personal best to be retained Line 3
    beta = None
    # proportion of global best to be retained Line 5
    delta = 0
    # proportion of the informants' best to be retained Line 4
    gamma = None
    # jump size of a particle   Line 6
    epsilon = 1
    # Dimension the size of plane that the swarm can move on and where the particles get their position from
    # might be a variable that allows the algorithm to simulate each particle moving in 3d space, which would require a change in position allowing for multiple
    dimension = 1

    particles = [Particle() for _ in range(swarmSize)]
    while count < iter:

        for particle in particles:  # Line 12
            #  finding the local best position and add
            if len(Best) == 0:
                Best.append(particle.position)  # Line 15
            elif np.any(fitness_Rastrign(particle.position) > fitness_Rastrign(Best[count])):  # Line 14
                Best.append(particle.position)  # Line 15

        for particle in particles:
            # variable that stores the previous fittest position of the local particle
            beta = Best  # Line 17

            # variable that stores the previous fittest position of informants of the Local particle position including the particle itself
            # Gamma might be the list of particle before their positions and velocity gets updated
            gamma = particles  # Line 18

            # variable that stores the previous global fittest position
            if delta == 0:
                delta = Best[count]  # Line 19
            elif np.any(fitness_Rastrign(Best[count]) > fitness_Rastrign(delta)):
                delta = Best[count]  # Line 19

            # uses right counter
            for i in range(dimension):  # Line 20
                # Random number from 0.0 to beta inclusive
                # your cognitive weight
               # b = np.random.uniform(0.0, (beta[i].position, dimension))  # Line 21
                b = np.random.uniform(0.0, (beta[count]))  # Line 21
                # Random number from 0.0 to gamma inclusive
                # your social weight component
                #c = np.random.uniform(0.0, (len(particles), dimension))  # Line 22
                #infomaxPos =
                c = np.random.uniform(0.0, (gamma[count].position))
                # Random number from 0.0 to delta inclusive
               # d = np.random.uniform(0.0, (len(particles), dimension))
                d = np.random.uniform(0.0, (delta))# Line 23
                # generates a new velocity for the particle
                # Pseudocode Line 24 compared to velocities = w * velocities + c1 * r1 * (best_positions - particles) + c2 * r2 * (swarm_best_position - particles)
                particle.velocity = alpha * particles[count].velocity + b * (beta[count] - particles[count].position) + c * (gamma[count].position - particles[count].position)  # Line 24

        for particle in particles:  # Line 25
            particle.position = particle.position + (epsilon * particle.velocity)  # Line 26
        count += 1
# particle that returns the best fitness at the end # might be delta the globalbest
    return delta

#--------------------------------------- FIX I AND COUNT FOR LOOPS
p1 = Particle()
f1 = fitness_Rastrign(p1.position)
print(p1.velocity)
print (f1)
Pso1 = PSO(10,10)
print(Pso1)