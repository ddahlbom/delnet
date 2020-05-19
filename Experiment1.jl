module Experiment1


using DelNetExperiment
using Plots

experimentname = "quicktest"

# Model Parameters
fs = 2000.0
num_neurons = 1000.0
p_contact = 0.1
p_exc = 0.8
maxdelay = 20.0
# tau_pre = 0.02
# tau_post = 0.02
# a_pre = 1.20
# a_post = 1.0
tau_post = 0.03125
tau_pre = 0.85*tau_post 
a_post = 0.0168 
a_pre = 0.0337 
synmax = 10.0
w_exc = 4.0
w_inh = -5.0

mp = ModelParams(fs, num_neurons, p_contact, p_exc, maxdelay,
				 tau_pre, tau_post, a_pre, a_post, synmax,
				 w_exc, w_inh)

# Trial Parameters
dur = 60.0
λ_noise = 3.0
randspikesize = 20.0
randinput = 1
inhibition = 1
inputmode = 2
inputweight = 20.0
recorddur = 30.0
recordstart = max(dur-recorddur, 0.0)
recordstop = dur
λ_instarttimes = 0.9

tp = TrialParams(dur, λ_noise, randspikesize, randinput, inhibition,
				 inputmode, inputweight, recordstart, recordstop, λ_instarttimes)


# Input parameters
λ_input = 50.0
inputdur = 0.1

input = genrandinput(λ_input, inputdur, 800)


# Run the experiment
numprocs=8
spikes, inputtimes = runnewexperiment(mp, tp, input, experimentname;
									  numprocs=numprocs)

# Quick analysis
p_spikes = spikeanalysisplot(spikes, input, inputtimes, recordstart, recordstop, mp.fs)

synapses = loadsynapses(experimentname);
p_syn = plot(synapses[1:800]; xlabel="Synapse Number", ylabel="Strength",
			 legend=:none)

p_img = pstplot(spikes, inputtimes, 2.0*inputdur, 1, Int(num_neurons), fs;
				xlabel="Time (s)", ylabel="Neuron Number")

spikeraster(p_img, input;
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
