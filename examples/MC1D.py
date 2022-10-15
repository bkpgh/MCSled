""" Test for mcsled """

import mcsled
import random
import copy
import time


class Particle1D:
    """ A 1D particle defined only by its position, x. """

    def __init__(self,x):
        self.x = x


class System:
    """ A system of N 1D particles.
    Has the required get_size, get_moves, and get_energy methods
    for mcsled.
    """

    def __init__(self,N,minx,maxx,ranseed=None):
        self.N = N
        self.minx = minx
        self.maxx = maxx
        self.saved_state = None
        self.use_dE = True
        if ranseed:
            random.seed(ranseed)

        # Randomly assign starting positions for N particles.
        startpositions = [random.random() * (maxx - minx) + minx
                          for i in range(self.N)]
        self.particles = [Particle1D(x) for x in startpositions]
        self.size = len(self.particles)

        # define moves
        self.jump = Move(1,0.3,self.get_particles())
        self.bigjump = Move(0.1,2.0,self.get_particles())
        self.moves = [self.jump,self.bigjump]

        # define energy components
        self.extE = ExternalEnergy(-4.0,4.0,100.0)
#        self.extE = ExternalEnergy(0.0,1.0,1.0)
        self.ppE = PPEnergy(0.05,0.3,1.0)

    def get_particles(self):
        """ Return the list of particle objects. """
        return self.particles

    def get_energy(self):
        """ Calculate and return the total energy of the system. """
        extE = sum([self.extE.E(p) for p in self.particles])
        mysum = 0.0
        for i,p in enumerate(self.particles[:-1]):
            otherps = self.particles[i + 1:]
            mye = sum([self.ppE.E(p,p2) for p2 in otherps])
            mysum += mye
        return mysum + extE

    def get_energy_change(self,move):
        """ Calculate and return energy change due to move. """
        p_moved = move.get_moved_p()
        p_unmoved = move.get_unmoved_p()
        pidx = self.particles.index(p_unmoved)
        dextE = self.extE.E(p_moved) - self.extE.E(p_unmoved)
        cutlist = self.particles[:pidx] + self.particles[pidx + 1:]
        mydE = (sum([self.ppE.E(p_moved,p2) for p2 in cutlist])
                - sum([self.ppE.E(p_unmoved,p2) for p2 in cutlist]))
        return dextE + mydE

    def get_moves(self):
        """ Return a list of the move objects. """
        return self.moves

    def get_size(self):
        """ Return the system size. """
        return self.size

    def save_state(self):
        """ Make a copy of the current state of the system. """
        self.saved_state  = copy.deepcopy(self.particles)

    def get_saved_state(self):
        """ Return the saved state. """
        return self.saved_state

    def set_saved_state(self):
        """ Set the saved state to the current state. """
        self.particles = copy.deepcopy(self.saved_state)


class Move:
    """ A move is a displacement of one particle by a random amount up to
    maxmove. 'probability' should be proportional to the fraction of moves
    chosen with this type.
    Has the required trial_move, make_move, unmake_move, and get_probability
    methods for mcsled.
    """

    def __init__(self,probability,maxmove,particles):
        self.probability = probability  # prop. to num times this move called
        self.maxmove = maxmove
        self.particles = particles
        self.p = None
        self.tmpp = Particle1D(0.0)
        self.dx = None

    def get_moved_p(self):
        return self.tmpp

    def get_unmoved_p(self):
        return self.p

    def trial_move(self):
        """ Chooses which particle to move and how far """
        self.p = random.choice(self.particles)
        self.dx = random.random() * (2 * self.maxmove) - self.maxmove
        self.tmpp.x = self.p.x + self.dx

    def make_move(self):
        """ Actually displace the input particle. """
        self.p.x += self.dx

    def unmake_move(self):
        """ Move it back in case of move rejection. """
        self.p.x -= self.dx

    def get_probability(self):
        """ Return the (unnormalized) probability for this move. """
        return self.probability


class ExternalEnergy:
    """ Convex external field. """

    def __init__(self,xleft,xright,strength):
        self.xleft = xleft
        self.xright = xright
        self.strength = strength
        self.n = 6

    def E(self,p):
        """ Return the particle-external field interaction energy. """
        if p.x < self.xleft or p.x > self.xright:
            return 1.0e6
        energy = self.strength * (1.0 / (p.x - self.xleft)**self.n
                                  + 1.0 / (p.x - self.xright)**self.n)
        return energy

# class ExternalEnergy:
#     """ Convex external field. """

#     def __init__(self,xmin,Emin,strength):
#         self.xmin = xmin
#         self.Emin = Emin
#         self.strength = strength
#         self.n = 2

#     def E(self,p):
#         """ Return the particle-external field interaction energy. """
#         return self.strength * (abs(p.x - self.xmin))**self.n + self.Emin


class PPEnergy:
    """ Square-well interaction between two particles. """

    def __init__(self,sigma1,sigma2,depth):
        self.sig1 = sigma1
        self.sig2 = sigma2
        self.depth = depth
        self.core = 1.0e6

    def E(self,p1,p2):
        """ Return the particle-particle energy
        for the input pair of particles.
        """
        dx = abs(p2.x - p1.x)
        if dx <= self.sig1:
            return(self.core)
        elif dx <= self.sig2:
            return(-self.depth)
        else:
            return(0.0)


if __name__ == "__main__":

    # Starting system is N 1D particles randomly distributed on
    # leftpos <= x <= rightpos
    N = 21        # Number of particles to create
    leftpos = -4  # Minimum position at which particles can be created
    rightpos = 4  # Maximum position at which particles can be created
    nout = 60     # number of character fields in simple 1D output

    def simpledump(nout,posl,posr):
        """ Simple output of 1D particle positions. """
        # nout is number of output character fields
        # posl and posr are the max left and right positions in units
        # of the particle positions.
        # A '*' is output at the position for each particle.
        # An 'x' is output at a position holding more than one particle.
        vis = [" "] * nout
        for p in sys.get_particles():
            relpos = int((p.x - posl) / (posr - posl) * nout)
            if relpos > nout - 1:
                relpos = nout - 1
            if relpos < 0:
                relpos = 0
            if vis[relpos] != " ":
                if vis[relpos] == "*":
                    n = 2
                else:
                    n = int(vis[relpos]) + 1
                vis[relpos] = str(n)
            else:
                vis[relpos] = "*"
        outstr = "".join(vis)
        print(outstr)

    print()
    print()

    # Create the system
    safety = 0.5
    sys = System(N,leftpos + safety,rightpos - safety,ranseed=9797973)
    print("Starting System: ")
    print("Energy = ",sys.get_energy())
    print()
    simpledump(nout,leftpos,rightpos)
    print()

    # Set up the Annealing schedule and the Simulation
    schedule = mcsled.AnnealingSchedule(Ti=10,Tf=0.001,reduce=0.96,
                                          ncycles=1000,nstop=25)
    sim = mcsled.MCSim(sys,schedule)

    # Run it.
    starttime = time.time()
    sim.anneal()
    endtime = time.time()
    deltatime = endtime - starttime
    print()
    print("The simulation took ",deltatime," seconds")

    # Dump the lowest energy state
    sys.set_saved_state()
    print("Best System at end: ")
    print("Energy = ",sys.get_energy())
    print()
    simpledump(nout,leftpos,rightpos)
