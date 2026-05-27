from pysim import PySim

import pprofile

profiler = pprofile.Profile()
with profiler:
    for i in range(1):
        PySim(nsegs=401).compute_impedance(ntrap=16, engine="accelerated")

# You can also write the result to the console:
profiler.print_stats()
profiler.dump_stats("profiler_stats.txt")
