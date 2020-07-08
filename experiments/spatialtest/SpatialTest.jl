module SpatialTest 

using DelNetExperiment
using Plots
using Measures
using JLD
using Random: seed!
using Match

# seed!(10)

modelname = "twotones"
trialname1 = "training"
trialname2 = "testing"

# Model Parameters
fs = 2000.0
a_exc = 0.02
d_exc = 8.0
a_inh = 0.1
d_inh = 2.0
v_default = -65.0
u_default = -13.0

synmax = 10.0
w_exc = 6.0
w_inh = -5.0

tau_pre = 0.02
tau_post = 0.02
a_pre = 1.2/10
a_post = 1.0/10
## I think these are from Masquelier, Thorpe et al.
# tau_post = 0.03125
# tau_pre = 0.85*tau_post 
# a_post = 0.0168 * 10.0
# a_pre = 0.0337 * 10.0

# Neuron Types 
type_exc = SimpleNeuronType("rs")
type_inh = SimpleNeuronType("fs")

# Make the delay graph
dims = (0.1, 0.1, 3.0)
types = [type_exc, type_inh]
ρs = [40000.0, 10000.0]
λs = [0.5, 0.25]
vs = [100.0, 100.0]

pos, neurontypes, delgraph = genpatch(dims, types, ρs, λs, vs, fs;
									  probfactor = 0.5,
									  numslices = 10,
									  maxlen=5.0,
									  verbose=true)
numexc = length(filter(x -> x == type_exc, neurontypes)) 

# Make the synapse graph
weights = Dict(type_exc => 3.0900,
			   type_inh => -6.0)
syngraph = gensyn(delgraph, neurontypes, weights)


# Set up neurons
idcs_exc = findall(x -> x == type_exc, neurontypes)
idcs_inh = findall(x -> x == type_inh, neurontypes)
neurons = [SimpleNeuron(t) for t ∈ neurontypes]

# Set up model parameter structure
num_neurons = Float64(length(neurons))
p_contact = sum(map(x -> x != 0 ? 1 : 0, delgraph))/(num_neurons*num_neurons)
p_exc = length(idcs_exc) /num_neurons
maxdelay = 1000.0*maximum(delgraph)/fs

p_contact_exc = sum(map(x -> x != 0 ? 1 : 0, delgraph[1:numexc,:])) /
				(num_neurons*numexc)
p_contact_inh = sum(map(x -> x != 0 ? 1 : 0, delgraph[numexc+1:end,:])) /
				(num_neurons*(num_neurons-numexc))
println("Maximum Delay Length: $maxdelay (ms)")
println("Percent Excitable: $p_exc")
println("Probability of contact (exc): $p_contact_exc")
println("Probability of contact (inh): $p_contact_inh")

mp = ModelParams(fs, num_neurons, p_contact, p_exc, maxdelay,
				 a_exc, d_exc, a_inh, d_inh, v_default, u_default,
				 synmax, w_exc, w_inh,
				 tau_pre, tau_post, a_pre, a_post)

# Training trial Parameters
dur = 10.0
recorddur = 10.0
λ_noise = 0.1
randspikesize = 00.0
randinput = 1
inhibition = 1
inputmode = 2
multiinputmode = 1
inputweight = 20.0
recordstart = max(0.0,dur-recorddur)
recordstop = dur 
λ_instarttimes = 0.5 
inputrefractorytime = 0.008 + 5*maxdelay/fs

tp1 = TrialParams(dur, λ_noise, randspikesize, randinput, inhibition,
				  inputmode, multiinputmode, inputweight, recordstart,
				  recordstop, λ_instarttimes, inputrefractorytime)


# Generate Inputs 
times = vcat([[0.0, 0.03] for _ ∈ 1:10]...)
input1 = channelscatter(times, 1:40)
inputs = [input1]

inputdur = times[end]



# Make the model
model = Model(mp, delgraph, syngraph, neurons)

# Run the training
numprocs = 10
@time results_trial = runexperiment("new", modelname, model, tp1, inputs;
								 	numprocs=numprocs,
								 	execloc="/home/dahlbom/research/delnet/")


## Quick analysis

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

p_spikes = spikeanalysisplot(#results_trial.output,
							 sortedoutput,
							 inputs,
							 results_trial.inputtimes,
							 results_trial.inputids,
							 results_trial.tp.recordstart,
							 results_trial.tp.recordstop,
							 mp.fs;
							 numneurons=Int(num_neurons),
							 windowdur=0.02)

p_syn = plot(filter(x->x>=0, [s.strength for s ∈ results_trial.synapses]);
			 xlabel="Synapse Number", ylabel="Strength",
			 legend=:none)

heatmaps = DelNetExperiment.pstplots(results_trial.output,
								  results_trial.inputtimes,
								  results_trial.inputids,
								  inputdur+5*maxdelay/1000.0,
								  1,
								  Int(num_neurons),
								  fs;
								  xlabel="Time (s)",
								  ylabel="Neuron Number")

for (k,input) ∈ enumerate(results_trial.input)
	spikeraster(heatmaps[k], input;
				markersize=4.0,
				markercolor=:white,
				markershape=:x,
				markerstrokewidth=0.0,
				legend=:none)
end


p1 = plot(p_spikes, p_syn, layout=@layout [a{0.93h}; b])
p2 = plot(heatmaps..., margins=2mm)

save("twotoneresults.jld", "results", results_trial)


end
