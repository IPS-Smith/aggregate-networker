work TBD

gen-aggregates is provided only for use with the inhouse CG sugars project
needs to be run from a directory within the simulation directory
i.e. simulation/analysis

need to redefine path to aggregate_networker in the gen-aggregates script

aggregate_networker currently has 1us traj length and 2000ps nstxout-compressed hardcoded, along with a 2000ps analysis timestep

These can be modified if needed, but will be generalised in the future

further help can be found by calling:

python aggregate_networker-edit.py --help
