module TestExperiment

using Random
using Plots

include("./julia/SpikePlot.jl")
include("./ExperimentTools.jl")
include("inputs.jl")

using .ExperimentTools


################################################################################
# Main Script
################################################################################
fs = 2000.0
num_neurons = 1000
p_contact = 0.1
p_exc = 0.8
maxdelay = 20.0
tau_pre = 0.02
tau_post = 0.02
a_pre = 1.20
a_post = 1.0
synmax = 10.0
w_exc = 4.0
w_inh = -5.0

mp = ModelParams(fs, num_neurons, p_contact, p_exc, maxdelay,
				 tau_pre, tau_post, a_pre, a_post, synmax, w_exc, w_inh)

dur = 2.5 
lambda = 3.0
randspikesize = 00.0
randinput = 1 	# <-- don't think this is doing anything anymore; check.
inhibition = 1
numinputs = 20
inputmode = 2
recordstart = 0 
recordstop  = dur 
lambdainput = 4 

tp = TrialParams(dur, lambda, randspikesize, randinput, inhibition, numinputs,
				 inputmode, recordstart, recordstop, lambdainput)


# Input Parameters
#f_in = 5.0
#input = periodicspiketrain(f_in, 1.0, mp.fs) .* 20.0
Random.seed!(5)
λ_in  = 35.0
intimes = sparsepoisson(λ_in, 0.100)
inspikes = channelscatter(intimes, 1:Int(round(num_neurons*p_exc)))


# Write Configuration Files
trialname = "$(Int(round(λ_in)))Hz_$(Int(round(tp.dur)))s_fs$(Int(round(mp.fs)))"
saveinput(inspikes, trialname)
writemparams(mp, trialname)
writetparams(tp, trialname)

# MPI Parameters
numprocs = 8

# Run Simulation
run(`mpirun -np $(numprocs) ./runtrial-mpi 0 $(trialname * "_mparams.dat") $(trialname * "_tparams.dat") $(trialname * "_input.bin") $(trialname)`)

# Open spikes and plot
spikes = SpikePlot.loadtrialspikesmpi(trialname)
inputstarttimes = SpikePlot.loadvector(trialname * "_instarttimes.txt")
p = inputrasterlines(recordstart, recordstop, spikes, inspikes, inputstarttimes, mp.fs)

end
