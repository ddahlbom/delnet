module TestExperiment

using DelNetExperiment
using Plots

modelname = "testmodel"
trialname1 = "training"
trialname2 = "testing"

# Model Parameters
fs = 2000.0
num_neurons = 1000.0
p_contact = 0.1
p_exc = 0.8
maxdelay = 20.0
tau_pre = 0.02
tau_post = 0.02
a_pre = 1.20
a_post = 1.0
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
dur = 5.0
λ_noise = 3.0
randspikesize = 20.0
randinput = 1
inhibition = 1
inputmode = 3
inputweight = 20.0
recordstart = 0
recordstop = dur
λ_instarttimes = 0.75

tp1 = TrialParams(dur, λ_noise, randspikesize, randinput, inhibition,
				  inputmode, inputweight, recordstart, recordstop, λ_instarttimes)


# Testing trial Parameters
recorddur = dur
recordstart = max(dur-recorddur, 0.0)
recordstop = dur 

tp2 = TrialParams(dur, λ_noise, randspikesize, randinput, inhibition,
				  inputmode, inputweight, recordstart, recordstop, λ_instarttimes)


# Input parameters
#λ_input = 3.0 * 800
λ_input = 10.0 
inputdur = 0.100

times = DelNetExperiment.sparsepoisson(λ_input, inputdur)
# input = DelNetExperiment.channeldup(times, [1,2,3,4,5,500,501,502,503,504] )
input = DelNetExperiment.channelscatter(times, 1:800)

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

graph[801:end,801:end] .= 0

# Run the training
numprocs=8
results_trial = runexperiment("new", modelname, mp, graph, tp1, input;
								 numprocs=numprocs,
								 execloc="/home/dahlbom/research/delnet/")

println("Training complete. Testing...")

# Run the trial

# results_trial = runexperiment("resume", modelname, trialname2, tp2, input;
# 							  numprocs=numprocs,
# 							  execloc="/home/dahlbom/research/delnet/")


# Quick analysis
p_spikes = spikeanalysisplot(results_trial.output,
							 results_trial.input,
							 results_trial.inputtimes,
							 results_trial.tp.recordstart,
							 results_trial.tp.recordstop,
							 mp.fs)

p_syn = plot(results_trial.synapses[1:800]; xlabel="Synapse Number", ylabel="Strength",
			 legend=:none)

p_img = pstplot(results_trial.output,
				results_trial.inputtimes,
				inputdur+5*maxdelay/1000.0,
				1,
				1000,
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
