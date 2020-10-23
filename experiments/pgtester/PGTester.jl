module PGTester

using DelNetExperiment
using Plots
using Measures
using JLD
using Random: seed!
using Match

# seed!(10)

modelname = "PGTester1"
trialname1 = "training"
trialname2 = "testing"

# Model Parameters
fs = 2000.0
num_neurons = 1000.0
p_contact = 0.10
p_exc = 0.80
maxdelay = 20.0

a_exc = 0.02
d_exc = 8.0
a_inh = 0.1
d_inh = 2.0
v_default = -65.0
u_default = -13.0

synmax = 10.0
J_exc = 45.00
J_inh = -60.0
w_exc = J_exc/√(p_contact*num_neurons)
w_inh = J_inh/√(p_contact*num_neurons)

a_pre  = 1.0*10
a_post = 1.2*10
tau_pre = 0.02
tau_post = (a_pre/a_post)*tau_pre

## I think these are from Masquelier, Thorpe et al.
# tau_post = 0.03125
# tau_pre = 0.85*tau_post 
# a_post = 0.0168 * 10.0
# a_pre = 0.0337 * 10.0

mp = ModelParams(fs, num_neurons, p_contact, p_exc, maxdelay,
				 synmax, tau_pre, tau_post, a_pre, a_post)

# Training trial Parameters
dur = 600.0 ; doplotting = true
recorddur = 10.0
λ_noise = 3.0
randspikesize = 20.0
randinput = 1
inhibition = 1
inputmode = 4
multiinputmode = 1
inputweight = 20.0
recordstart = max(0.0,dur-recorddur)
recordstop = dur 
λ_instarttimes = 0.5 
inputrefractorytime = 0.008 + 5*maxdelay/fs

tp1 = TrialParams(dur, λ_noise, randspikesize, randinput, inhibition,
				  inputmode, multiinputmode, inputweight, recordstart, recordstop,
				  λ_instarttimes, inputrefractorytime)



# Make the neurons
num_exc = Int(round(p_exc * num_neurons))
num_inh = 1000 - num_exc 
neurontypes = vcat([SimpleNeuronType("rs") for _ ∈ 1:num_exc],
			 	   [SimpleNeuronType("fs") for _ ∈ 1:num_inh])
type_exc = SimpleNeuronType("rs")
type_inh = SimpleNeuronType("fs")
numexc = length(filter(n->n==type_exc,neurontypes))
neurons = [SimpleNeuron(t) for t ∈ neurontypes]
idcs_exc = findall(x -> x == type_exc, neurontypes)
idcs_inh = findall(x -> x == type_inh, neurontypes)

# Generate the delay graph
delgraph = zeros(Int64,Int(num_neurons),Int(num_neurons))
possibledelays = 1:1:20  # in ms
possibledelays = Int64.(fs .* (possibledelays ./ 1000.0))
p_contact_inh = 1.125*p_contact 
# for r ∈ 1:Int(num_neurons)
# 	for c ∈ 1:Int(num_neurons)
# 		if rand() < p_contact && r != c
# 			if neurontypes[r] == type_inh && neurontypes[c] == type_inh
# 				delgraph[r,c] = 0	
# 			else	
# 				if neurontypes[r] == type_exc
# 					delgraph[r,c] =  rand(possibledelays) 
# 				elseif neurontypes[r] == type_inh
# 					delgraph[r,c] = 1
# 				end
# 			end
# 		end
# 	end
# end
for r ∈ 1:Int(num_neurons)
	for c ∈ 1:Int(num_neurons)
		if neurontypes[r] == type_inh
			if neurontypes[c] == type_inh
				delgraph[r,c] = 0	
			else
				if rand() < p_contact_inh && r != c
					delgraph[r,c] = 1
				end
			end
		else	
			if rand() < p_contact && r != c
				delgraph[r,c] =  rand(possibledelays) 
			end
		end
	end
end

# Generate the synapse graph
syngraph = zeros(Float64, size(delgraph))
for r ∈ 1:Int(num_neurons)
	w = neurontypes[r] == type_exc ? w_exc : w_inh
	for c ∈ 1:Int(num_neurons)
		syngraph[r,c] = delgraph[r,c] != 0 ? w : 0.0
	end
end

# Generate Inputs 
numexc = Int(round(p_exc*num_neurons))

f0 = 10.0
sigdur = 0.5 
numchannels = Int64(round(num_neurons/5))
p = 0.1
numspikes = Int(round(sigdur*f0))
intimes = [k/f0 for k ∈ 0:numspikes-1]
input1 = stochasticblock(intimes, p, idcs_exc[1:numchannels])
input2 = stochasticblock(intimes, p, idcs_exc[1:numchannels])


inputs = [input1, input2]

inputdur = sigdur 

# Make the model
model = Model(mp, delgraph, syngraph, neurons)

# Run the training
numprocs = 8
@time results_trial = runexperiment("new", modelname, model, tp1, inputs;
								 	numprocs=numprocs,
								 	execloc="/home/dahlbom/research/delnet/")


# Sort output
spikes_exc = Array{Spike,1}(undef,0)
for s ∈ filter(s->s.n ∈ idcs_exc, results_trial.output) 
	push!(spikes_exc, Spike( findfirst(x->x==s.n, idcs_exc), s.t) )
end
spikes_inh = Array{Spike,1}(undef,0)
for s ∈ filter(s->s.n ∈ idcs_inh, results_trial.output) 
	push!(spikes_inh, Spike(findfirst(x->x==s.n, idcs_inh)+numexc, s.t))
end
sortedoutput = vcat(spikes_exc, spikes_inh)

# Sort input
inputs = Array{Array{Spike,1},1}(undef,0)
for input ∈ results_trial.input
	spikes_input = Array{Spike,1}(undef,0)
	for s ∈ input 
		push!(spikes_input, Spike( findfirst(x->x==s.n, idcs_exc), s.t))
	end
	push!(inputs, spikes_input)
end


# Synapse analyses
synapses = filter(s -> neurontypes[s.source+1] == type_exc,
				  results_trial.synapses)
delays, strengths = DelNetExperiment.averagedelaylength(synapses)

hist = histogram([s.strength for s ∈ results_trial.synapses])

# Plotting
if doplotting
	p_spikes = spikeanalysisplot(#results_trial.output,
								 sortedoutput,
								 inputs,
								 results_trial.inputtimes,
								 results_trial.inputids,
								 results_trial.tp.recordstart,
								 results_trial.tp.recordstop,
								 mp.fs;
								 numneurons=Int(num_neurons),
								 windowdur=0.001)
	p_syn = bar(delays, strengths)
	plot!(p_syn, delays, w_exc .* ones(length(strengths)); ls=:dash)
	p1 = plot(p_spikes, p_syn, layout=@layout [a{0.93h}; b])
end


end
