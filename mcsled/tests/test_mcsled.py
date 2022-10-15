"""
Unit tests for the mcsled package.
"""

import importlib
import sys
import mcsled
import pytest
pytestmark = pytest.mark.unit


def test_mcsled_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "mcsled" in sys.modules


def test_annealing_schedule():
    print()
    myti = 10.0
    mytf = 1.0
    myreduce = 0.9
    myncycles = 10
    mynstop = 5
    sched = mcsled.AnnealingSchedule(Ti=myti,
                                     Tf=mytf,
                                     reduce=myreduce,
                                     ncycles=myncycles,
                                     nstop=mynstop)

    assert sched.Ti == myti
    assert sched.Tf == mytf
    assert sched.reduce == myreduce
    assert sched.ncycles == myncycles
    assert sched.nstop == mynstop

    sched = mcsled.AnnealingSchedule()

    assert sched.Ti == 100.0
    assert sched.Tf == 0.01
    assert sched.reduce == 0.95
    assert sched.ncycles == 100
    assert sched.nstop == 25


class Mymove:
    def get_probability(self):
        return 1

    def trial_move(self):
        return 1

    def make_move(self):
        return 1

    def unmake_move(self):
        return 1


class Mysystem:

    def __init__(self):
        self.use_dE = False
        self.E = 0.0
        self.dE = 0.0

    def get_moves(self):
        return [Mymove()]

    def get_energy(self):
        return self.E

    def get_energy_change(self,imove):
        return self.dE

    def get_size(self):
        return 1


randcount = -1


def myrandom():
    global randcount
    nums = [0.2, 0.3, 0.4, 0.4, 0.51, 0.6, 0.8, 0.9]
    randcount += 1
    if randcount >= 8:
        return 0.9
    else:
        # print("Fake random number {}: {}".format(randcount,nums[randcount]))
        return nums[randcount]


@pytest.fixture
def get_test_obj():
    sled = mcsled
    sched = sled.AnnealingSchedule()
    mysys = Mysystem()
    obj = sled.MCSim(mysys,sched)
    mcsled.random.random = myrandom
    return obj,mysys,sled


def test_decide(get_test_obj):
    print()

    obj,junk,junk2 = get_test_obj
    T = 1.0
    dE = 0.0
    decision = obj.decide(dE,T)
    assert decision is True
    dE = 1.0e30
    decision = obj.decide(dE,T)
    assert decision is False
    dE = 1.0
    decision = obj.decide(dE,T)
    assert decision is True  # e^-1 ~ 0.3678 > 0.3
    decision = obj.decide(dE,T)
    assert decision is False  # e^-1 ~ 0.3678 < 0.4


def test_choose_move(get_test_obj):
    print()
    obj,junk,junk2 = get_test_obj
    obj.cummoveprobabilities = [0.5, 1.0]
    imove = obj.choose_move()
    assert imove == 0
    imove = obj.choose_move()
    assert imove == 1


def decide_false(x,y):
    return False


def decide_true(x,y):
    return True


def test_mc_step(get_test_obj):
    print()
    obj,mysys,junk2 = get_test_obj
    # Test mc_step energy paths
    T = 1.0
    mysys.use_dE = False

    Enew = 1.0
    mysys.E = -1.0
    obj.decide = decide_true
    Enewnew = obj.mc_step(T,Enew)
    assert Enewnew == -1.0

    Enew = 1.0
    mysys.E = 2.0
    obj.decide = decide_false
    Enewnew = obj.mc_step(T,Enew)
    assert Enewnew == 1.0

    # Test mc_step change-in-energy paths
    mysys.use_dE = True

    Enew = 1.0
    mysys.dE = 1.0
    obj.decide = decide_false
    obj.energy_change = mysys.get_energy_change
    Enewnew = obj.mc_step(T,Enew)
    assert Enewnew == 1.0

    Enew = 1.0
    mysys.dE = 1.0
    obj.decide = decide_true
    obj.energy_change = mysys.get_energy_change
    Enewnew = obj.mc_step(T,Enew)
    assert Enewnew == 2.0


def test_mc_block(get_test_obj):
    print()
    obj,mysys,junk2 = get_test_obj
    # Test mc_block energy paths
    T = 1.0
    nsteps = 3
    mysys.use_dE = True

    Enew = 1.0
    mysys.dE = 1.0
    obj.lowE = Enew
    obj.decide = decide_true
    obj.energy_change = mysys.get_energy_change
    Enewnew = obj.mc_block(nsteps,T,Enew)
    assert Enewnew == 4.0
    assert obj.lowE == 1.0

    Enew = 1.0
    mysys.dE = 1.0
    obj.lowE = Enew
    obj.decide = decide_false
    obj.energy_change = mysys.get_energy_change
    Enewnew = obj.mc_block(nsteps,T,Enew)
    assert Enewnew == 1.0
    assert obj.lowE == 1.0

    Enew = 1.0
    mysys.dE = -1.0
    obj.lowE = Enew
    obj.decide = decide_true
    obj.energy_change = mysys.get_energy_change
    Enewnew = obj.mc_block(nsteps,T,Enew)
    assert Enewnew == -2.0
    assert obj.lowE == -2.0


def test_check_stop():
    print()
    sled = mcsled
    mysys = Mysystem()
    sched = sled.AnnealingSchedule(Ti=1.0,Tf=0.2,reduce=0.5,ncycles=3,nstop=5)
    obj = sled.MCSim(mysys,sched)
    obj.Thistory = [1.0] * 12
    obj.Eblockhistory = [1.0] * 12

    T = 1.0
    nstop = 5
    obj.lowE = 1.0
    stopanneal = obj.check_early_stop(12,T,nstop)
    assert stopanneal is True

    obj.lowE = 2.0
    stopanneal = obj.check_early_stop(12,T,nstop)
    assert stopanneal is False


def test_anneal():
    print()
    mytiny = 1.0e-8
    sled = mcsled
    sched = sled.AnnealingSchedule(Ti=1.0,Tf=0.2,reduce=0.5,ncycles=3,nstop=10)
    mysys = Mysystem()
    obj = sled.MCSim(mysys,sched)

    mysys.use_dE = True
    mysys.E = 1.0
    mysys.dE = -1.0
    obj.lowE = 1.0
    obj.decide = decide_true
    obj.energy_change = mysys.get_energy_change
    obj.anneal()
    assert obj.lowE == -8.0
    assert (obj.Thistory[-1] - 0.25) < mytiny
