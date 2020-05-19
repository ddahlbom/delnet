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

dur = 1.0 
lambda = 3.0
randspikesize = 20.0
randinput = 1 	# <-- don't think this is doing anything anymore; check.
inhibition = 1
numinputs = 20
inputmode = 2
inputweight = 20.0
recordstart = max(dur-10.0, 0.0) 
recordstop  = dur 
lambdainput = 4 

tp = TrialParams(dur, lambda, randspikesize, randinput, inhibition, numinputs,
				 inputmode, inputweight, recordstart, recordstop, lambdainput)


# Input Parameters
#f_in = 5.0
#input = periodicspiketrain(f_in, 1.0, mp.fs) .* 20.0
numexc = Int(round(num_neurons*p_exc))
Random.seed!(5)
λ_in  = 35.0
intimes = sparsepoisson(λ_in, 0.100)
inspikes = channelscatter(intimes, 1:numexc)


# Write Configuration Files
# trialname = "$(Int(round(λ_in)))Hz_$(Int(round(tp.dur)))s_fs$(Int(round(mp.fs)))"
trialname = "TestTrial"
saveinput(inspikes, trialname)
writemparams(mp, trialname)
writetparams(tp, trialname)

# MPI Parameters
numprocs = 8

# Run Simulation
run(`mpirun -np $(numprocs) ./runtrial-mpi 0 $(trialname * "_mparams.txt") $(trialname * "_tparams.txt") $(trialname * "_input.bin") $(trialname)`)

#run(`mpirun -np $(numprocs) ./runtrial-mpi 1 $(trialname) $(trialname * "_tparams.txt") $(trialname * "_input.bin") $(trialname*"1")`)

# Open spikes and plot
spikes = loadspikes(trialname)
#spikes = loadspikes(trialname*"1")
inputstarttimes = SpikePlot.loadvector(trialname * "_instarttimes.txt")
#inputstarttimes = SpikePlot.loadvector(trialname * "1_instarttimes.txt")
p1 = inputrasterlines(recordstart, recordstop, spikes, inspikes, inputstarttimes, mp.fs)
synapses = loadsynapses(trialname)
p2 = plot(synapses[1:numexc], ylims=(3,5))

end
