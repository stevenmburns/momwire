from pysim import PySim
import cProfile
import pstats

with cProfile.Profile() as pr:
    for i in range(1):
        PySim(nsegs=4001).compute_impedance(ntrap=16, engine="accelerated")

    ps = pstats.Stats(pr).sort_stats(pstats.SortKey.TIME)
    ps.dump_stats("foo")
    ps.print_stats()
