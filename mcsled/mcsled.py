""" MCSled: Monte Carlo Simulated Annealing.
    You need to provide two objects as input arguments to MCSim object.

    1. An object (see commented System template) describing your system
    containing methods:
        A: get_moves() which returns a list of objects each of which
        (see commented Move class template)
        has these methods:
            (1) trial_move(), which provides parameters for the move;
            perhaps indexes of which parts of the system to modify.
            (2) make_move(),  which actually makes the change.
            (3) unmake_move(), which reverses the change
            when a move is rejected.
            (4) get_probability(), which provides (unnormalized)
            probability for that move.
        These methods are all called without arguments, so their parent object
        must handle their state and any required inputs.

        B: get_energy() method which returns the total energy of the system,
        given its current state. Also called without arguments.

        C: a get_size() method which returns a measure of the system size.
        In the MC simulation, each cycle is defined as a number of moves
        equal to the system size. Also called without arguments.

        D: Optionally, a save_state() method which saves a copy
        of the current state as an attribute of the input system object.
        This will be used to hold the lowest energy state found during
        the MC simulation.

        E: Optionally, a get_energy_change(move) method which calculates
        the change in energy due to a trial move object rather than calculating
        the energy of the whole system. This is sometimes much faster
        that calculating the entire energy at each step. To use this routine,
        also set an attribute of the system object, use_dE, to be True.
        The get_energy_change routine will be passed the chosen move object
        by the annealing simulation.


    2. An instance of the AnnealingSchedule class (defined in this module)
        which holds the parameters for the simulation:
        Ti, Tf, reduce, ncyles, nstop.
"""

import random
import math
import multiprocessing as mp


class Replicates():
    """ Implement multiple simulations on multicore machine.
    system = System object containing get_energy etc. methods.
    schedule = AnnealingSchedule object.
    nproc = [optional] number of processors for parallel runs.
        if integer, taken as number of processes to run
        if float, taken as multiplier on total number of available processors.
        if missing, # proc used will be 85% of those available, rounded down.
    """

    def __init__(self,system,schedule,nproc=None,ranseed=None):
        if nproc is None:
            np = int(0.85 * mp.cpu_count())
        elif isinstance(nproc,int):
            np = nproc
        elif isinstance(nproc,float):
            np = int(nproc * mp.cpu_count())
        self.np = np
        print("Replicates will use {} threads.".format(np))
        self.system = system
        self.schedule = schedule
        if ranseed is not None:
            random.seed(ranseed)

    def run(self):
        """ Sets up the pool of worker processes and runs the annealing method
        in each.
        """
        # each anneal_parallel method returns the system object after the run
        pool = mp.Pool(processes=self.np)
        # These ranseeds do NOT seem to get the child processes to have
        # consistent ran number sequences in different runs.
        ranseeds = [int(random.random() * 1000000) for i in range(self.np)]
        print("Random number seeds: ")
        for r in ranseeds:
            print(r)
        results = [pool.apply_async(MCSim(self.system,self.schedule,
                                    ranseed=ranseeds[i]).
                                    anneal_parallel,args=(i,))
                   for i in range(self.np)]
        output = [p.get() for p in results]
        return output


class MCSim:

    tiny = 1.0e-16

    def __init__(self,system,schedule,ranseed=None):
        self.lowE = 1.0 / self.tiny
        self.Ti = schedule.Ti
        self.Tf = schedule.Tf
        self.reduce = schedule.reduce
        self.ncycles = schedule.ncycles
        self.nstop = schedule.nstop
        self.system = system
        self.moves = system.get_moves()
        self.energy = system.get_energy
        if ranseed is not None:
            random.seed(ranseed)
        if system.use_dE:
            self.energy_change = system.get_energy_change
        self.size = system.get_size()
        # convert move weights/propabilities to cumulative probabilities
        mysum = 0.0
        mysum = sum([move.get_probability() for move in self.moves])
        probabilities = [move.get_probability() / mysum for move in self.moves]
        self.cummoveprobabilities = [sum(probabilities[0:i + 1])
                                     for i in range(len(probabilities))]
        self.Thistory = []
        self.Eblockhistory = []

    def choose_move(self):
        """ chooses next move type based on cumulative probabilities """
        x = random.random()
        for i,p in enumerate(self.cummoveprobabilities):
            if x <= p:
                return i

    def decide(self,dE,T):
        """ Returns T or F for Metropolis acceptance criterion """
        if dE <= 0.0:
            return True
        x = random.random()
        if x <= math.exp(- dE / T):
            return True
        else:
            return False

    def mc_step(self,T,Enew):
        """ one step of the Monte Carlo simulation """
#        print("-------------top of mc_step------------------")

        Eold = Enew
        imove = self.choose_move()
        moveobj = self.moves[imove]  # get the move
        moveobj.trial_move()  # set the move parameters

        if self.system.use_dE:
            dE = self.energy_change(moveobj)
        else:
            self.moves[imove].make_move()   # actually change the system
            Enew = self.energy()
            dE = Enew - Eold

        decision = self.decide(dE,T)
        if decision:
            #  accepted
            if self.system.use_dE:
                self.moves[imove].make_move()   # actually change the system
                Enew = Enew + dE
            else:
                pass  # already changed
        else:
            # rejected
            if self.system.use_dE:
                pass  # do nothing, system not changed
            else:
                Enew = Eold
                self.moves[imove].unmake_move()  # change the system back

#        print("-------------bottom of mc_step------------------")

        self.Enew = Enew
        return Enew

    def mc_block(self,nsteps,T,Enew):
        """ A block of Monte Carlo: runs for nsteps number of steps
        at T Temperature. Assumes system energy Enew is correct on input.
        """
#        print("-------------top of block------------------")
#        print("T = ",T,"     Enew = ",Enew)

        for istep in range(nsteps):
            Enew = self.mc_step(T,Enew)

            if Enew < self.lowE:
                self.lowE = Enew
                try:
                    self.system.save_state()
                except Exception:
                    pass

#        print("-------------end of block------------------")

        return Enew

    def anneal_parallel(self,junk):
        """ This exists just so multiprocessing methods can hand it an arg. """
        return self.anneal()

    def anneal(self,threshold=0):
        """  Run an annealing simulation consisting of multiple Monte Carlo
        blocks run at decreasing temperatures according to the input schedule.
        """
        print("------------------- Anneal Start --------------------------")

        Tinit = self.Ti
        Tfinal = self.Tf
        reduce = self.reduce
        ncycles = self.ncycles
        nstop = self.nstop

        iblock = 0
        Enew = self.energy()

        try:
            self.system.save_state()  # save current as lowest energy state
        except Exception:
            pass

        # A step is one move, A block is ncycles * self.size moves
        nsteps = self.size * ncycles
        T = Tinit
        while (T >= Tfinal):

            iblock += 1
            Enew = self.mc_block(nsteps,T,Enew)
            # print("E = ",Enew)
            T = T * reduce

            if self.check_early_stop(iblock,T,nstop,threshold=threshold):
                print("Anneal stopping: Change in lowest Energy over ",
                      "last {} MC blocks".format(nstop),
                      " is less than threshold",threshold)
                break

        print("------------------- Anneal Finish -------------------------")
        return self.system

    def check_early_stop(self,iblock,T,nstop,threshold=0):
        """ Build temperature and low-energy histories and check
        if low energy has not changed in nblock steps. If it has,
        exit the annealing run.
        """
        self.Thistory.append(T)
        self.Eblockhistory.append(self.lowE)

        if iblock > nstop:
            stopanneal = True
            nlen = len(self.Eblockhistory)
            E = self.Eblockhistory[-1]
            # print("-------------------------------------------------")
            for i in range(nlen - 1, nlen - nstop - 1, -1):
                test = (abs(E - self.Eblockhistory[i]) <= threshold)
                stopanneal = (stopanneal and test)
            # if stopanneal:
            #     for i in range(nlen - 1, nlen - nstop - 1, -1):
            #         print("T = ",self.Thistory[i],
            #              "E = ",self.Eblockhistory[i])
            return stopanneal


class AnnealingSchedule:
    """ Holds annealing schedule parameters. Only Geometric Cooling for now.
        Ti: Initial Temperature
        Tf: Final Temperature
        reduce: Factor by which temperature is reduced between MC blocks
        ncycles: Number of cycles (= one move for each thing in simulation)
        nstop: Stop simulation if low energy hasn't changed for nstop cycles.
    """

    def __init__(self,Ti=100.0,Tf=0.01,reduce=0.95,ncycles=100,nstop=25):
        self.Tf = Tf
        self.Ti = Ti
        self.reduce = reduce
        self.ncycles = ncycles
        self.nstop = nstop

# =============================================================================
# class System:
#     """ System object template with required methods.
#     """
#
#     def __init__(self):
#         self.use_dE = False...
#
#     def get_energy(self):
#         """ Return the system energy as a function of its state. """
#         return ...
#
#     def get_moves(self):
#         """ Return list of Move objects """
#         return ...
#
#     def get_size(self):
#         """ Return some measure system size; e.g. number of basic elements.
#         """
#         return ...
#
#     def save_state(self):
#         """ Optional: Save the current state of the system as a copy. """
#         self.saved_state  = ...
#
#     def get_energy_change(self,move):
#     """ Calculate and return energy change due to move. """
#         return dE ...
#
# =============================================================================

# =============================================================================
# class Move:
#     """ Move object template with required methods.
#     """
#
#     def __init__(self):
#
#     def trial_move(self):
#         """ Find parameters for an instance of this move type """
#
#     def make_move(self):
#         """ Change the system to implement this move type. """
#
#     def unmake_move(self):
#         """ Change the system back to how it was before make_move. """
#
#     def get_probability(self):
#         """ Return something proportional to the fraction of times this
#         move should be called.
#         """
# =============================================================================


if __name__ == "__main__":
    pass
