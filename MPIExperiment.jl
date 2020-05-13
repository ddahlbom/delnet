module MPIExperiment

using Plots
include("./julia/SpikePlot.jl")

struct ModelParams
	fs::Float64
	num_neurons::UInt64
	p_contact::Float64
	p_exc::Float64
	maxdelay::Float64
	tau_pre::Float64
	tau_post::Float64
	a_pre::Float64
	a_post::Float64
	synmax::Float64
	w_exc::Float64
	w_inh::Float64
end
ModelParams() = ModelParams(2000.0, 1000, 0.1, 0.8, 20.0, 0.02, 0.02, 1.20,
							1.0, 10.0, 4.0, -5.0)

struct TrialParams
	dur::Float64
	lambda::Float64
	randspikesize::Float64
	randinput::Bool
	inhibition::Bool
	numinputs::UInt64
	inputmode::UInt64
	recordstart::Float64
	recordstop::Float64
	lambdainput::Float64
end
TrialParams() = TrialParams(1.0, 3.0, 20.0, 1, 1, 100, 1, 0.0, 1.0, 0.5)

################################################################################
# Functions
################################################################################
function spiketrain(f, dur, fs)
	N_pat = Int(round(dur * fs))
	dn_pat = Int(round(fs/f))
	train = zeros(N_pat)
	idx = 1
	while idx <= N_pat
		train[idx] = 1.0
		idx += dn_pat
	end
	train
end


function saveinput(input, trialname)
	open(trialname * "_input.bin", "w") do f
		write(f, length(input)) 	# write an Int64 (long int)
		write(f, input) 			# write an array of Float64 (double)
	end
end

function writemparams(m::ModelParams, trialname)
	open(trialname * "_mparams.dat", "w") do f
		write(f, "fs\t\t\t$(m.fs)\n")
		write(f, "num_neurons\t\t$(Float64(m.num_neurons))\n")
		write(f, "p_contact\t\t$(m.p_contact)\n")
		write(f, "p_exc\t\t\t$(m.p_exc)\n")
		write(f, "maxdelay\t\t$(m.maxdelay)\n")
		write(f, "tau_pre\t\t\t$(m.tau_pre)\n")
		write(f, "tau_post\t\t$(m.tau_post)\n")
		write(f, "a_pre\t\t\t$(m.a_pre)\n")
		write(f, "a_post\t\t\t$(m.a_post)\n")
		write(f, "synmax\t\t\t$(m.synmax)\n")
		write(f, "w_exc\t\t\t$(m.w_exc)\n")
		write(f, "w_inh\t\t\t$(m.w_inh)\n")
	end
end

function writetparams(t::TrialParams, trialname)
	open(trialname * "_tparams.dat", "w") do f
		write(f, "dur\t\t\t$(t.dur)\n")
		write(f, "lambda\t\t\t$(t.lambda)\n")
		write(f, "randspikesize\t\t$(t.randspikesize)\n")
		write(f, "randinput\t\t$(Float64(t.randinput))\n")
		write(f, "inhibition\t\t$(Float64(t.inhibition))\n")
		write(f, "numinputs\t\t$(t.numinputs)\n")
		write(f, "inputmode\t\t$(t.inputmode)\n")
		write(f, "recordstart\t\t$(t.recordstart)\n")
		write(f, "recordstop\t\t$(t.recordstop)\n")
		write(f, "lambdainput\t\t$(t.lambdainput)\n")
	end
end

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
randinput = 1
inhibition = 1
numinputs = 100
inputmode = 1
recordstart = 0.0
recordstop = 1.0
lambdainput = 0.5

tp = TrialParams(dur, lambda, randspikesize, randinput, inhibition, numinputs,
				 inputmode, recordstart, recordstop, lambdainput)


# Override above for test
mp = ModelParams()
tp = TrialParams()

# Input Parameters
f_in = 5.0
input = spiketrain(f_in, 1.0, mp.fs) .* 20.0


# Write Configuration Files
trialname = "$(Int(round(f_in)))Hz_$(Int(round(tp.dur)))s_fs$(Int(round(mp.fs)))"
saveinput(input, trialname)
writemparams(mp, trialname)
writetparams(tp, trialname)

# MPI Parameters
numprocs = 4

# Run Simulation
run(`mpirun -np $(numprocs) ./runtrial-mpi 0 $(trialname * "_mparams.dat") $(trialname * "_tparams.dat") $(trialname * "_input.bin") $(trialname)`)


# Open spikes and plot
spikes = SpikePlot.loadtrialspikesmpi(trialname)
p = SpikePlot.inputraster(0, tp.dur, spikes, input, mp.fs)

end
