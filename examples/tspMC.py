""" Test for mcsled """

import mcsled
import random
import copy
import time
import math
import matplotlib.pyplot as plt
import functools


class City:
    """ A city defined only by its position, x,y. """

    def __init__(self,x):  # x should be a list with two elements: x&y postions
        self.x = x


class System:
    """ A system of N Cities and a cyclic path between them.
    Has the required get_size, get_moves, and get_energy methods
    for mcsled.
    """

    def __init__(self,N,cities,use_dE):
        # minx and max x are 2-element lists
        self.cities = cities
        self.N = len(cities)
        self.saved_state = None
        self.moves = None
        self.use_dE = use_dE

        # define moves
        self.swap = SwapMove(1,self.get_cities())
        self.moves = [self.swap]

        # define energy components
        self.energy = Energy()

    def get_cities(self):
        """ Return the list of city objects. """
        return self.cities

    def get_energy(self):
        """ Calculate and return the total energy of the system. """
        energy = self.energy.E(self.cities)
        return energy

    def get_energy_change(self,move):
        """ Calculate and return energy change due to move. """
        dE = self.energy.dE(self.cities,move)
        return dE

    def get_moves(self):
        """ Return a list of the move objects. """
        return self.moves

    def get_size(self):
        """ Return the system size. """
        return self.N

    def save_state(self):
        """ Make a copy of the current state of the system. """
        self.saved_state  = copy.deepcopy(self.cities)

    def get_saved_state(self):
        """ Return the saved state. """
        return self.saved_state

    def set_saved_state(self):
        """ Set the saved state to the current state. """
        self.cities = copy.deepcopy(self.saved_state)


class Move:
    """ A move is a modification of the cities path.
    Has the required trial_move, make_move, unmake_move, and get_probability
    methods for mcsled.
    """

    def __init__(self,probability,cities):
        self.probability = probability  # prop. to num times this move called
        self.cities = cities

    def get_probability(self):
        """ Return the (unnormalized) probability for this move. """
        return self.probability


class SwapMove(Move):
    """ A swap move trades places for two cities in the path.
    'probability' should be proportional to the fraction of moves
    chosen with this type.
    Has the required trial_move, make_move, unmake_move, and get_probability
    methods for mcsled.
    """

    def __init__(self,probability,cities):
        super().__init__(probability,cities)
        self.c1 = None
        self.c2 = None

    def get_moved_cities(self):
        return self.c1, self.c2

    def trial_move(self):
        """ Chooses two different cities """
        self.c1 = random.choice(self.cities)
        self.c2 = random.choice(self.cities)
        while self.c2 == self.c1:
            self.c2 = random.choice(self.cities)

    def make_move(self):
        """ Actually swap the cities in the path """
        self.idx1 = self.cities.index(self.c1)
        self.idx2 = self.cities.index(self.c2)
        self.cities[self.idx1] = self.c2
        self.cities[self.idx2] = self.c1

    def unmake_move(self):
        """ Move them back in case of move rejection. """
        self.cities[self.idx1] = self.c1
        self.cities[self.idx2] = self.c2


class Energy:
    """ The path length for the traversing the cities. """

    def E(self,cities):
        distances = [self.mydist(cities[i],cities[i + 1])
                     for i in range(len(cities) - 1)]
        totald = sum(distances) + self.mydist(cities[0],cities[-1])
        return totald

    @functools.lru_cache(maxsize=16384)
    def mydist(self,c1,c2):
        return(math.dist(c1.x,c2.x))

    def dE(self,cities,move):
        c1,c2 = move.get_moved_cities()
        c1left,c1right = self.get_neighbors(c1,cities)
        c2left,c2right = self.get_neighbors(c2,cities)
        if c1 is c2left:  # c2 = c1right also
            oldpartialE = self.mydist(c1left,c1) + self.mydist(c2right,c2)
            newpartialE = self.mydist(c1left,c2) + self.mydist(c2right,c1)
        elif c1 is c2right:  # c2 = c1left also
            oldpartialE = self.mydist(c1right,c1) + self.mydist(c2left,c2)
            newpartialE = self.mydist(c1right,c2) + self.mydist(c2left,c1)
        else:
            oldpartialE = (self.mydist(c1left,c1) + self.mydist(c1right,c1)
                           + self.mydist(c2left,c2) + self.mydist(c2right,c2))
            newpartialE = (self.mydist(c1left,c2) + self.mydist(c1right,c2)
                           + self.mydist(c2left,c1) + self.mydist(c2right,c1))
        deltaE = newpartialE - oldpartialE
        return deltaE

    def get_neighbors(self,city,cities):
        N = len(cities)
        idx = cities.index(city)
        left = idx - 1
        right = idx + 1
        if idx == 0:
            left = N - 1
        elif idx == N - 1:
            right = 0
        return cities[left], cities[right]


if __name__ == "__main__":

    def dump(cities,fignum,figname,txt):
        txt = str(txt)
        x = [city.x[0] for city in cities]
        x.append(x[0])
        y = [city.x[1] for city in cities]
        y.append(y[0])
        plt.figure(fignum)
        plt.plot(x, y)
        plt.scatter(x, y)
        plt.text(0,0,txt)
        plt.savefig(figname)

    # Starting system is N randomly distributed cities
    N = 71        # Number of cities to create
    minx = lowerleftpos = [0,0]
    mapsize = 100
    maxx = upperrightpos = [mapsize,mapsize]

    # Randomly assign positions for N cities.
    # If the Python random number generator does not change,
    # the best solution to this problem found so far is E = 680.35571.
    ranseed = 9797973
    random.seed(ranseed)
    x = [random.random() * (maxx[0] - minx[0]) + minx[0]
         for i in range(N)]
    y = [random.random() * (maxx[1] - minx[1]) + minx[1]
         for i in range(N)]
    cities = [City([a,b]) for a,b in zip(x,y)]

    # If use_dE then system will use the energy_change method
    # rather than the full energy method
    use_dE = True

    # If use_parallel, then will use multiprocessing to replicate the system
    # over several cores
    use_parallel = True

    # Create the system
    sys = System(N,cities,use_dE)
    print("Starting System: ")
    energy = sys.get_energy()
    print("Energy = ",energy)
    cities = sys.get_cities()
    dump(cities,0,"Start",energy)

    # Set up the Annealing schedule and the Simulation
    schedule = mcsled.AnnealingSchedule(Ti=10,Tf=0.001,reduce=0.99,
                                        ncycles=500,nstop=25)

    if use_parallel:
        ranseed = int(random.random() * 100000)
        sims = mcsled.Replicates(sys,schedule,ranseed=ranseed)
    else:
        sim = mcsled.MCSim(sys,schedule)

    # Run it.
    starttime = time.time()
    if use_parallel:
        parallel_output = sims.run()
    else:
        sim.anneal()
    endtime = time.time()
    deltatime = endtime - starttime
    print()
    print("The simulation took ",deltatime," seconds")

    # Dump the lowest energy state

    if use_parallel:
        lowE = 1e16
        for i,xsys in enumerate(parallel_output):
            xsys.set_saved_state()
            E = xsys.get_energy()
            print("E = ",E)
            cities = xsys.get_cities()
            dump(cities,i + 1,"Final" + str(i),E)
            if E < lowE:
                lowE = E
                bestcities = cities
        dump(bestcities,i + 2,"Best",lowE)
        print("Lowest Energy over parallel runs:",lowE)
    else:
        sys.set_saved_state()
        print("Best System at end: ")
        E = sys.get_energy()
        print("Energy = ",E)
        cities = sys.get_cities()
        dump(cities,1,"Best",E)
