module Consonants

using DelNetExperiment
using Plots
using Measures
using JLD
using Random: seed!
using Match

# seed!(10)

modelname = "consonants"
trialname1 = "training"
trialname2 = "testing"

# Model Parameters
fs = 4000.0
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
w_exc = 5.25
w_inh = -5.0

tau_pre = 0.02
tau_post = 0.02
a_pre = 1.2
a_post = 1.0

## I think these are from Masquelier, Thorpe et al.
# tau_post = 0.03125
# tau_pre = 0.85*tau_post 
# a_post = 0.0168 * 10.0
# a_pre = 0.0337 * 10.0

mp = ModelParams(fs, num_neurons, p_contact, p_exc, maxdelay,
				 synmax, tau_pre, tau_post, a_pre, a_post)

# Training trial Parameters
dur = 1500.0; plotting = false
recorddur = 500.0
λ_noise = 0.1
randspikesize = 00.0
randinput = 1
inhibition = 1
inputmode = 2
multiinputmode = 1
inputweight = 20.0
recordstart = max(0.0,dur-recorddur)
recordstop = dur 
λ_instarttimes = 1.0 
inputrefractorytime = 0.25 

tp1 = TrialParams(dur, λ_noise, randspikesize, randinput, inhibition,
				  inputmode, multiinputmode, inputweight, recordstart, recordstop,
				  λ_instarttimes, inputrefractorytime)



# Make the neurons
# neurontypes = vcat([SimpleNeuronType("rs") for _ ∈ 1:800],
# 			 	   [SimpleNeuronType("fs") for _ ∈ 1:200])
type_exc = SimpleNeuronType("rs")
type_inh = SimpleNeuronType("fs")
randvals = rand(1000)
neurontypes = map( n -> n < p_exc ? type_exc : type_inh, randvals)
numexc = length(filter(n->n==type_exc,neurontypes))
# neurontypes = vcat([type_exc for _ ∈ 1:800], [type_inh for _ ∈ 1:200])
neurons = [SimpleNeuron(t) for t ∈ neurontypes]
idcs_exc = findall(x -> x == type_exc, neurontypes)
idcs_inh = findall(x -> x == type_inh, neurontypes)

# Generate the delay graph
delgraph = zeros(Int64,Int(num_neurons),Int(num_neurons))
possibledelays = 1:Int(round(fs*maxdelay/1000.0)) # b/c maxdel in ms
for r ∈ 1:Int(num_neurons)
	for c ∈ 1:Int(num_neurons)
		if rand() < p_contact && r != c
			if neurontypes[r] == type_inh && neurontypes[c] == type_inh
				delgraph[r,c] = 0	
			else	
				if neurontypes[r] == type_exc
					delgraph[r,c] =  rand(possibledelays) 
				elseif neurontypes[r] == type_inh
					delgraph[r,c] = 1 
				end
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
# tonedur = 0.05
# numharmonics = 5
numchannels = 30
numperchan = 10 
densityfactor = 10000.0/5.5 #1.23
refractorytime = 0.005

numinstances = 5

kas = [loadaudiofile("/home/dahlbom/research/delnet/experiments/vowelsandconsonants/audio/ka0$k.wav", fs) for k ∈ 1:numinstances]

sas = [loadaudiofile("/home/dahlbom/research/delnet/experiments/vowelsandconsonants/audio/ba0$k.wav", fs) for k ∈ 1:numinstances]

kas_spikes = [auditoryspikes(tone, numchannels, numperchan, idcs_exc, fs;
							 rt = refractorytime,
							 densityfactor = densityfactor)
			  for tone ∈ kas]

sas_spikes = [auditoryspikes(tone, numchannels, numperchan, idcs_exc, fs;
							 rt = refractorytime,
							 densityfactor = densityfactor)
			  for tone ∈ sas]

inputs = [kas_spikes..., sas_spikes...]

inputdur = maximum( vcat([length(x) for x ∈ kas],
						 [length(x) for x ∈ sas]) ) / fs
println("Input duration: $inputdur")

# numinstances = 2 
# 
# tone133 = [hc(133.0, numharmonics, tonedur, fs) for _ ∈ 1:numinstances]
# spikes133 = [auditoryspikes(tone, numchannels, numperchan, idcs_exc, fs;
# 						   rt=refractorytime,
# 						   densityfactor=densityfactor)
# 			 for tone ∈ tone133]
# tone200 = [hc(200.0, numharmonics, tonedur, fs) for _ ∈ 1:numinstances]
# spikes200 = [auditoryspikes(tone, numchannels, numperchan, idcs_exc, fs;
# 						   rt=refractorytime,
# 						   densityfactor=densityfactor)
# 			 for tone ∈ tone200]
# 
# inputs = [spikes133..., spikes200...]
# 
# inputdur = tonedur


# Make the model
model = Model(mp, delgraph, syngraph, neurons)

# Run the training
numprocs = 8
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

if plotting
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

	p1 = plot(p_spikes, p_syn, layout=@layout [a{0.93h}; b])


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
	p2 = plot(heatmaps..., margins=2mm)
end

save("consonantsresults.jld", "results", results_trial)


end
