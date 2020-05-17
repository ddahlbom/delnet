module ExperimentTools

using Random
using Plots
include("./julia/SpikePlot.jl")

include("inputs.jl")


export ModelParams, TrialParams, writemparams, writetparams 

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
	inputweight::Float64
	recordstart::Float64
	recordstop::Float64
	lambdainput::Float64
end
TrialParams() = TrialParams(1.0, 3.0, 20.0, 1, 1, 100, 1, 0.0, 1.0, 0.5)

################################################################################
# Functions
################################################################################
function writemparams(m::ModelParams, trialname)
	open(trialname * "_mparams.txt", "w") do f write(f, "fs\t\t\t$(m.fs)\n")
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
	open(trialname * "_tparams.txt", "w") do f
		write(f, "dur\t\t\t$(t.dur)\n")
		write(f, "lambda\t\t\t$(t.lambda)\n")
		write(f, "randspikesize\t\t$(t.randspikesize)\n")
		write(f, "randinput\t\t$(Float64(t.randinput))\n")
		write(f, "inhibition\t\t$(Float64(t.inhibition))\n")
		write(f, "numinputs\t\t$(t.numinputs)\n")
		write(f, "inputmode\t\t$(t.inputmode)\n")
		write(f, "inputweight\t\t$(t.inputweight)\n")
		write(f, "recordstart\t\t$(t.recordstart)\n")
		write(f, "recordstop\t\t$(t.recordstop)\n")
		write(f, "lambdainput\t\t$(t.lambdainput)\n")
	end
end

end
