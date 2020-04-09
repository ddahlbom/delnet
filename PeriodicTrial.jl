module PeriodicTrial

using Plots
include("../dahlpoly/SpikePlot.jl")

struct ModelParams
	fs::Float64
	num_neurons::Float64
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
ModelParams() = ModelParams(2000.0, 1000.0, 0.1, 0.8, 20.0, 0.02, 0.02, 1.20,
							1.0, 10.0, 3.0, -4.0)

struct TrialParams
	fs::Float64
	dur::Float64
	lambda::Float64
	randspikesize::Float64
	randinput::Bool
	inhibition::Bool
	numinputs::UInt32
	inputidcs::Array{UInt32, 1}
end
# Ptr{Cuint} UInt32
TrialParams() = TrialParams(2000.0, 2.5, 3.0, 20.0, 1, 1, 100, [])

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
		write(f, "num_neurons\t\t$(m.num_neurons)\n")
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
		write(f, "fs\t\t\t$(t.fs)\n")
		write(f, "dur\t\t\t$(t.dur)\n")
		write(f, "lambda\t\t\t$(t.lambda)\n")
		write(f, "randspikesize\t\t$(t.randspikesize)\n")
		write(f, "randinput\t\t$(Float64(t.randinput))\n")
		write(f, "inhibition\t\t$(Float64(t.inhibition))\n")
		write(f, "numinputs\t\t$(t.numinputs)\n")
	end
end
################################################################################
# Main Script
################################################################################
# Model Parameters
fs = 2000
n = 1000
p_contact = 0.1
p_exc = 0.8
maxdelay = 20.0
tau_pre = 0.02
tau_post = 0.02
a_pre = 1.20
a_post = 1.0
synmax = 10.0
w_exc = 3.0
w_inh = -4.0

mp = ModelParams(fs, n, p_contact, p_exc, maxdelay, tau_pre, tau_post, a_pre,
				 a_post, synmax, w_exc, w_inh) 

# Trial Parameters
dur = 1.00
λ = 3.0 				# noise density
randspikesize = 20.0 	# noise magnitude
randinput = 1
inhibition = 1
numinputs = 70

tp = TrialParams(fs, dur, λ, randspikesize, randinput, inhibition, numinputs, [])
# Input Parameters
f_in = 10.0
input = spiketrain(f_in, 1.00, 2000.0) .* 20.0


# Write Configuration Files
trialname = "$(Int(round(f_in)))Hz_1000s"
saveinput(input, trialname)
writemparams(mp, trialname)
writetparams(tp, trialname)

# Run Simulation
run(`./runtrial-exec $(trialname * "_mparams.dat") $(trialname * "_tparams.dat") $(trialname * "_input.bin") $(trialname)`)

# Open spikes and plot
spikes = SpikePlot.loadspikes(trialname*"_spikes.dat"; timetype=Float64)
p = SpikePlot.inputraster(0, dur, spikes, input, fs)

end
