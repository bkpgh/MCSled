""" Function Minimization Test for mcsled """

import mcsled
import random
import math
import time
import copy


class System:
    """ An x,y point and an almost easy function to minimize.
    Minimum is 0 at x,y = 0,0. Adding a local minumum.
    Has the required get_size, get_moves, and get_energy methods
    for mcsled.
    """

    def __init__(self,minxy,maxxy,initx,inity,ranseed=None):
        self.position = [initx,inity]
        self.oldx = None
        self.oldy = None
        self.minx = minxy
        self.maxx = maxxy
        self.saved_state = None
        self.use_dE = False
        self.size = 1000
        if ranseed:
            random.seed(ranseed)

        # define moves
        self.jump = Move(1,0.3,self.minx,self.maxx,self.position)
        self.bigjump = Move(0.1,2.0,self.minx,self.maxx,self.position)
        self.moves = [self.jump,self.bigjump]

    def get_size(self):
        """ Size is arbitrary for a simple one-point function minimization.
        self.size hard-wired in __init__
        """
        return self.size

    def get_moves(self):
        """ Return a list of the move objects. """
        return self.moves

    def function(self,x,y):
        """  quadratic plus local well """
        f = x**2 + y**2
        if x < -2.3 and x > -2.7:
            if y < 2.7 and y > 2.3:
                f -= 8.0
        return f

    def get_energy(self):
        f = self.function(self.position[0],self.position[1])
        # print("In get_energy x,y,f: ",self.position[0],self.position[1],f)
        return f

    def get_state(self):
        return self.position

    def save_state(self):
        """ Make a copy of the current state of the system. """
        self.saved_state  = copy.deepcopy(self.position)

    def get_saved_state(self):
        """ Return the saved state. """
        return self.saved_state

    def set_saved_state(self):
        """ Set the saved state to the current state. """
        self.position = copy.deepcopy(self.saved_state)


class Move:
    """ A move is a displacement of the test point by a random amount up to
    maxmove. 'probability' should be proportional to the fraction of moves
    chosen with this type.
    Has the required trial_move, make_move, unmake_move, and get_probability
    methods for MCHammer.
    """

    def __init__(self,probability,maxmove,minxy,maxxy,system_position):
        self.probability = probability  # prop. to num times this move called
        self.maxmove = maxmove
        self.minxy = minxy
        self.maxxy = maxxy
        self.pos = system_position
        self.dx = None
        self.dy = None

    def trial_move(self):
        self.dx = random.random() * (2 * self.maxmove) - self.maxmove
        self.dy = random.random() * (2 * self.maxmove) - self.maxmove
        # print("trial dx,dy: ",self.dx,self.dy)

    def make_move(self):
        """ Actually displace the point. """
        self.pos[0] += self.dx
        if self.pos[0] > self.maxxy:
            self.pos[0] = self.maxxy
        if self.pos[0] < self.minxy:
            self.pos[0] = self.minxy
        self.pos[1] += self.dy
        if self.pos[1] > self.maxxy:
            self.pos[1] = self.maxxy
        if self.pos[1] < self.minxy:
            self.pos[1] = self.minxy
        # print("displaced point: ",self.pos[0],self.pos[1])

    def unmake_move(self):
        """ Move it back in case of move rejection. """
        self.pos[0] -= self.dx
        self.pos[1] -= self.dy

    def get_probability(self):
        """ Return the (unnormalized) probability for this move. """
        return self.probability


if __name__ == "__main__":

    xinit = -2.5
    yinit = 2.5
    minxy = -5.0
    maxxy = 5.0

    # Create the system
    safety = 0.5
    sys = System(minxy,maxxy,xinit,yinit,ranseed=9797973)
    print("Starting System: ")
    (x,y) = sys.get_state()
    print("x,y = ",x,y)
    print("Energy = ",sys.get_energy())
    print()

    # Set up the Annealing schedule and the Simulation
    schedule = mcsled.AnnealingSchedule(Ti=10.0,Tf=0.001,reduce=0.96,
                                        ncycles=200,nstop=25)
    sim = mcsled.MCSim(sys,schedule)

    # Run it.
    starttime = time.time()
    threshold = 1.0e-4
    sim.anneal(threshold)
    endtime = time.time()
    deltatime = endtime - starttime
    print()
    print("The simulation took ",deltatime," seconds")

    # Dump the lowest energy state
    sys.set_saved_state()
    print("Best System at end: ")
    (x,y) = sys.get_state()
    print("x,y = ",x,y)
    print("Energy = ",sys.get_energy())
    print()
