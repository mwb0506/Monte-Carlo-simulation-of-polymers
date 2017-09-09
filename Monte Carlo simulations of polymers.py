import time  # Adding timer using a clock
import matplotlib.pyplot as plt  # Basic plotting
import numpy as np  # Numerical calculations
from scipy.spatial.distance import cdist

# Note that the outdated PEP8 80 character restriction is not abided by but instead the more standard 144 character
# limit of Github and this editor (PyCharm Community Edition) is chosen

start_time = time.clock()  # Start time for global timer
loop_time = start_time  # Start time for loop timer

# Variables
n = 250  # Amount of beads
T = 1  # Temperature in terms of epsilon/kb
steps = 6  # Amount of different angles
pol = 10000  # Amount of polymers
prune_fraction = 0.01  # Fraction of polymers to be deleted by pruning/enhance (max=0.33)

# Constants
epsilon = 0.25  # As given in 10.6
sigma = 0.8  # As given in 10.6

# Allocation
pos = np.zeros((pol, n, 2), dtype=float)  # Positions of the atoms for every polymer
pol_weight = np.ones((pol, n), dtype=float)  # Weights for pruned-enriched Rosenbluth method (PERM)
mean_gyr = np.zeros(n, dtype=float)  # Average radius of gyration per polymer length
std_gyr = np.zeros(n, dtype=float)  # Standard deviation of average radius of gyration per polymer length
mean_energy = np.zeros(n, dtype=float)  # Average polymer energy per polymer length
std_energy = np.zeros(n, dtype=float)  # Standard deviation of average polymer energy per polymer length


# Initialization
pos[:, 1, 0] = 1  # First atom at (0,0), second atom at (1,0)
prune = round(prune_fraction * pol)  # Amount of polymers to be deleted by pruning/enhanced


def pot(radius):
    """
    Calculates the energy of the Lennard-Jones potential based on the squared radius
    :param radius: Radius
    :return: Potential energy
    """
    potential = 4 * epsilon * ((sigma / radius) ** 12 - (sigma / radius) ** 6)
    return potential


def perm(polymer_weight):
    """
    The pruned-enriched Rosenbluth method (PERM) algorithm
    :param polymer_weight: The polymer weight of the last added bead
    :return: Updated polymer weights
    """
    rank = polymer_weight.argsort()  # Sorted indexes of the polymer weights
    np.random.shuffle(rank[:2 * prune])  # Shuffle (randomize) which ones to deleted/pruned
    polymer_weight[rank[prune:2 * prune]] *= 2  # Double polymer weight for half of the pruned polymers
    pos[rank[:prune], :, :] = pos[rank[pol - prune:], :, :]  # Replace the pruned for the enhanced
    polymer_weight[rank[pol - prune:]] /= 2  # Half the weight of the enhanced polymers
    polymer_weight[rank[:prune]] = polymer_weight[rank[pol - prune:]]  # Replace weight of the pruned
    return polymer_weight


def add_bead(atom, polymer_weight):
    """
    Adds a new bead to every polymer and updates their weights
    :param atom: Which bead should be added
    :param polymer_weight: Polymer weights of the last bead for every polymer
    :return: Returns polymer weight for every polymer
    """
    angle = np.repeat(np.transpose([np.random.uniform(0, 2 * np.pi / steps, size=pol)]), steps, axis=1)  # Offset
    angle += np.repeat([np.arange(steps) * 2 * np.pi / steps], pol, axis=0)  # Set of angles for each offset
    w = np.empty((pol, steps), dtype=float)  # Allocate weight matrix
    for polymer in range(pol):  # For each polymer
        new_bead = np.repeat(pos[polymer, atom - 1, :][:, np.newaxis], steps, axis=1)  # Last bead location
        new_bead += [np.cos(angle[polymer, :]), np.sin(angle[polymer, :])]  # Add possible new beads
        radii = cdist(np.transpose(new_bead), pos[polymer, :atom, :])  # All radii between new possible beads and old
        w[polymer, :] = np.exp(-np.sum(pot(radius=radii), axis=1) / T)  # Calculate weights
    total_weight = np.sum(w, axis=1)  # Sum of the weights
    w /= np.transpose(np.repeat([total_weight], steps, axis=0))  # Turn into probabilities
    w_choice = np.cumsum(w, axis=1)  # Turn into cumulative probabilities
    c = np.random.rand(pol)  # Choose a random number for each polymer
    choice = np.argmax(w_choice > np.repeat(c[:, np.newaxis], steps, axis=1), axis=1)  # Choose angle index
    angle = np.choose(choice, angle.T)  # Angle corresponding with angle index
    pos[:, atom, 0] = pos[:, atom - 1, 0] + np.cos(angle)  # Add x-coordinates new bead
    pos[:, atom, 1] = pos[:, atom - 1, 1] + np.sin(angle)  # Add y-coordinates new bead

    polymer_weight *= total_weight  # Update weights
    max_polymer_weight = np.max(polymer_weight)  # Most likely polymer
    polymer_weight /= max_polymer_weight  # Weights now relative to most likely polymer
    return polymer_weight


def end_dist(position, polymer_weight, bootstrap=pol):
    """
    Calculates the end-to-end distance and its standard deviation via bootstrap
    :param position: Positions of the atoms for every polymer
    :param polymer_weight: Polymer weights of the entire polymer
    :param bootstrap: Amount of bootstrapping, defaults to amount of polymers
    :return: Mean and standard deviation of end-to-end distance for every available polymer length
    """
    rsq = np.square(np.linalg.norm(position, axis=2))  # Squared radius
    mean_distance = np.sum(polymer_weight * rsq, axis=0) / np.sum(polymer_weight, axis=0)  # Expectancy of R^2
    weighted_sum = polymer_weight * rsq  # Numerator of <R^2>
    ind = np.random.randint(pol, size=bootstrap)  # Choose indices to bootstrap over
    std_distance = np.std(weighted_sum[ind, :] / np.sum(polymer_weight[ind, :], axis=0), axis=0)  # Standard deviation
    return mean_distance, std_distance


def radius_of_gyration(position, polymer_weight, bootstrap=pol):
    """
    Calculates the radius of gyration 
    :param position: Positions of the atoms for every polymer
    :param polymer_weight: Polymer weights of the entire polymer
    :param bootstrap: Amount of bootstrapping, defaults to amount of polymers
    :return: Mean and standard deviation of bead to center of mass distance for every available polymer
    """
    mass_center = np.sum(position, axis=1) / position.shape[1]  # Center of mass of polymers
    mc_matrix = np.swapaxes(np.repeat(mass_center[:, :, np.newaxis], position.shape[1], axis=2), 1,
                            2)  # center of mass values with same dimensions as position
    sq_cmd = np.square(
        np.linalg.norm(position - mc_matrix, axis=2))  # Squared distance of beads from center of mass per polymer
    mean_sq_cmd = np.sum(sq_cmd, axis=1) / position.shape[1]  # Mean distance of beads from center of mass per polymer
    weighted_sum = polymer_weight[:, position.shape[1] - 1] * mean_sq_cmd  # Numerator of <R^2>
    mean_distance = np.sum(weighted_sum, axis=0) / np.sum(polymer_weight[:, position.shape[1] - 1],
                                                          axis=0)  # Mean distance of the beads from center of mass
    ind = np.random.randint(pol, size=bootstrap)  # Choose indices to bootstrap over
    std_distance = np.std(weighted_sum[ind] / np.sum(polymer_weight[ind, position.shape[1]-1]))  # Standard deviation
    return mean_distance, std_distance


def polymer_energy(position, polymer_weight, bootstrap=pol):
    """
    Calculates the average potential energy of a polymer and its standard deviation via bootstrap
    Note that due to its n^2 relation in speed, this severely bottlenecks the code if left on
    :param position: Positions of the atoms for every polymer
    :param polymer_weight: Polymer weights of the entire polymer
    :param bootstrap: Amount of bootstrapping, defaults to amount of polymers
    """
    energy = np.zeros(position.shape[0])  # Vector of energy per polymer
    for polymer in range(pol):  # For each polymer
        r_sq = cdist(position[polymer, :, :], position[polymer, :, :])  # Squared distance between beads
        r_sq[r_sq == 0] = 10 ** 10 # Removes potential energy of particle to itself
        energy[polymer] = np.sum(pot(r_sq))
    weighted_sum = energy * polymer_weight[:, position.shape[1] - 1]  # Numerator of <R^2>
    mean_pot_energy = np.sum(weighted_sum) / np.sum(polymer_weight[:, position.shape[1] - 1],
                                                axis=0)  # Mean energy of the polymers
    ind = np.random.randint(pol, size=bootstrap)  # Choose indices to bootstrap over
    std_pot_energy = np.std(weighted_sum[ind] / np.sum(polymer_weight[ind, position.shape[1]-1]))  # Standard deviation
    return mean_pot_energy, std_pot_energy


# Add beads up to n atoms
for i in range(2, n):
    pol_weight[:, i-1] = perm(polymer_weight=pol_weight[:, i-1])  # Update polymer weight before adding new bead
    pol_weight[:, i] = add_bead(atom=i, polymer_weight=pol_weight[:, i - 1])  # Add bead i
    mean_gyr[i], std_gyr[i] = radius_of_gyration(position=pos[:, :i+1, :],
                                                 polymer_weight=pol_weight)  # Mean/std radius of gyration
    mean_energy[i], std_energy[i] = polymer_energy(position=pos[:, :i+1, :], polymer_weight=pol_weight)
    print("Added bead %4.0f in %6.4f seconds" % (i + 1, time.clock() - loop_time))  # End time for loop timer
    loop_time = time.clock()  # Starting time for loop timer

mean_dist, std_dist = end_dist(position=pos, polymer_weight=pol_weight)  # Calculate mean/std end-to-end distance

print("Made %4.0f beads in %6.4f seconds" % (n, time.clock() - start_time))  # End time for global timer

# Saving variables to be plotted
np.save("mean_dist_%4.2f_T_%3.1f" % (prune_fraction, T), mean_dist)
np.save("std_dist_%4.2f_T_%3.1f" % (prune_fraction, T), std_dist)
np.save("mean_gyr_%4.2f_T_%3.1f" % (prune_fraction, T), mean_gyr)
np.save("std_gyr_%4.2f_T_%3.1f" % (prune_fraction, T), std_gyr)
np.save("mean_energy_%4.2f_T_%3.1f" % (prune_fraction, T), mean_energy)
np.save("std_energy_%4.2f_T_%3.1f" % (prune_fraction, T), std_energy)
np.save("position_%4.2f_T_%3.1f" % (prune_fraction, T), pos)


# Plot the end-to-end distance versus the polymer length
plt.figure(1)
plt.gca().set_xscale('log')
plt.gca().set_yscale('log')
plt.errorbar(np.arange(n), mean_dist, yerr=std_dist)
plt.show(block=False)

# Plot polymers to check for crossings
for i in range(10):
    plt.figure(i+2)
    plt.plot(np.transpose(pos[i-2, :, 0]), np.transpose(pos[i-2, :, 1]), '-o')
    plt.show(block=False)
plt.show()







