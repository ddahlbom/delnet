module ClusteredInput

using DelNetExperiment
using Plots
using Random: seed!

seed!(1234)

modelname = "clusteredinput"
trialname1 = "training"
trialname2 = "testing"

# Model Parameters
fs = 10000.0
num_neurons = 1000.0
p_contact = 0.1
p_exc = 0.80
maxdelay = 20.0
tau_pre = 0.02
tau_post = 0.02
a_pre = 1.2*10
a_post = 1.0*10
# tau_post = 0.03125
# tau_pre = 0.85*tau_post 
# a_post = 0.0168 * 10.0
# a_pre = 0.0337 * 10.0
synmax = 10.0
w_exc = 4.0
w_inh = -5.0

mp = ModelParams(fs, num_neurons, p_contact, p_exc, maxdelay,
				 tau_pre, tau_post, a_pre, a_post, synmax,
				 w_exc, w_inh)

# Training trial Parameters
dur = 100.0
recorddur = 10.0
λ_noise = 3.0
randspikesize = 20.0
randinput = 1
inhibition = 1
inputmode = 2
inputweight = 20.0
recordstart = max(0.0,dur-recorddur)
recordstop = dur 
λ_instarttimes = 1.0 
inputrefractorytime = 0.008 + 5*maxdelay/fs

tp1 = TrialParams(dur, λ_noise, randspikesize, randinput, inhibition,
				  inputmode, inputweight, recordstart, recordstop,
				  λ_instarttimes, inputrefractorytime)



# Input parameters
numexc = Int(round(p_exc*num_neurons))
λ_input = 1000.0 

#times = DelNetExperiment.sparserefractorypoisson(λ_input, inputdur, 0.004)
times = [0.000, 0.002, 0.003, 0.008]
dn = 20

inputdur = maximum(times)
spikes = [DelNetExperiment.channeldup([times[k]], ((k-1)*dn+1):(k*dn))
		  for k ∈ 1:length(times)]
input = vcat(spikes...)

# Generate the graph
graph = zeros(Int64,Int(num_neurons),Int(num_neurons))
possibledelays = 1:Int(round(fs*maxdelay/1000.0))
for i ∈ 1:Int(num_neurons)
	for j ∈ 1:Int(num_neurons)
		if rand() < p_contact && i != j
			graph[i,j] = i < p_exc*num_neurons ? rand(possibledelays) : 1
		end
	end
end

i = Int(round(p_exc*num_neurons))
graph[i+1:end,i+1:end] .= 0

# Run the training
numprocs=8
@time results_trial = runexperiment("new", modelname, mp, graph, tp1, input;
								 numprocs=numprocs,
								 execloc="/home/dahlbom/research/delnet/")


# Quick analysis
p_spikes = spikeanalysisplot(results_trial.output,
							 results_trial.input,
							 results_trial.inputtimes,
							 results_trial.tp.recordstart,
							 results_trial.tp.recordstop,
							 mp.fs;
							 numneurons=Int(num_neurons))

p_syn = plot(filter(x->x>=0, results_trial.synapses);
			 xlabel="Synapse Number", ylabel="Strength",
			 legend=:none)

p_img = pstplot(results_trial.output,
				results_trial.inputtimes,
				inputdur+5*maxdelay/1000.0,
				1,
				Int(num_neurons),
				fs;
				xlabel="Time (s)", ylabel="Neuron Number")

spikeraster(p_img, results_trial.input;
			markersize=4.0,
			markercolor=:white,
			markershape=:x,
			markerstrokewidth=0.0,
			legend=:none)

l_rc = @layout [ a{0.75h}; b]
p_rc = plot(p_img, p_syn, layout=l_rc)

l = @layout [ a{0.65w} b]
p = plot(p_spikes, p_rc, layout=l)

end
