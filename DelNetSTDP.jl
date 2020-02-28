module DelNetSTDP

using Plots
include("./DelNet.jl")
using .DelNet

# -------------------- Data Structures --------------------

# mutable struct NodeState
# 
# end

mutable struct Neuron
	v::Float64
	u::Float64
	a::Float64
	d::Float64
end


Neuron() = Neuron(-65.0, -13.0, 0.0, 0.0)

a_exc = 0.02
a_inh = 0.1
d_exc = 8.0
d_inh = 2.0

w_exc = 5.5
w_inh = -5.0

# τ_pre = 0.034
# A_pre = 51.0
# τ_post = 0.014
# A_post = 103.0
τ_pre = 0.020
A_pre = 0.12
τ_post = 0.020
A_post = 0.1

syn_max = 10.0

# -------------------- Parameters --------------------
fs = 1000.0
dt = 1/fs
dur = 0.5
ts = collect(0:dt:dur)
n_exc = 800
n_inh = 200
n = n_exc + n_inh 		# number of elements
p = 0.100
d_max = 20
verbose = false
noise = 0.0 			# probability of random firing
num_steps = 1000

# ---------- Generate Network, Nodes and Delay Lines --------------------

# Generate graph
g = blobgraph(n, p, collect(1:d_max))
g[n_exc+1:end,:] = map(x -> x != 0 ? 1 : 0, g[n_exc+1:end,:]) # inh delays = 1
dn = delnetfromgraph(g)

# Initialize neurons
neurons    = [Neuron() for _ ∈ 1:n]
synapses   = [Float64[] for _ ∈ 1:n]
trace_pre  = [Float64[] for _ ∈ 1:n_exc]
spike_pre  = [Float64[] for _ ∈ 1:n_exc]
trace_post = zeros(n_exc)
spike_post = zeros(n_exc)

for i ∈ 1:n_exc
	neurons[i].a = a_exc
	neurons[i].d = d_exc 
	synapses[i]  =  w_exc .* ones(dn.nodes[i].num_in)
	trace_pre[i] = zeros(dn.nodes[i].num_in)
	spike_pre[i] = zeros(dn.nodes[i].num_in)
end

for i ∈ n_exc+1:n
	neurons[i].a = a_inh
	neurons[i].d = d_inh
	synapses[i]  = w_inh .* ones(dn.nodes[i].num_in)
end



# -------------------- Run Simulation --------------------
spikes = [0.0 0.0]
rand_count = 0

synapse_w = zeros(length(ts))
syn_trace = zeros(length(ts))
neu_trace = zeros(length(ts))
potential = zeros(length(ts))

for (k, t) ∈ enumerate(ts)
	global dn, spikes, rand_count

	( (k%100) == 0 ) && ( println("Time: $(t+dt)") )
	# update node and calculate ouput, push into delnet
	for k ∈ 1:n

		# Get input from delay lines
		inval = 0.0
		inputs = getinputs(k,dn)
		inval += sum(synapses[k] .* inputs)

		# Update synapse traces
		if k <= n_exc
			for p ∈ 1:length(inputs)
				spike_pre[k][p] = inputs[p]
				#trace_pre[k][p] = min(1.0, trace_pre[k][p]*(1.0-(dt/τ_pre))+spike_pre[k][p])
				trace_pre[k][p] = trace_pre[k][p]*(1.0-(dt/τ_pre))+spike_pre[k][p]
			end
		end

		# Random input (noise)
		if rand() < 1.0/n
			inval += 20.0 * (fs/1000.0)
			rand_count += 1
		end

		# Update neuron state
		neurons[k].v += 500.0 * dt .* ((0.04 * neurons[k].v + 5.0) * neurons[k].v
								+ 140.0 - neurons[k].u + inval)
		neurons[k].v += 500.0 * dt .* ((0.04 * neurons[k].v + 5.0) * neurons[k].v
								+ 140.0 - neurons[k].u + inval)
		neurons[k].u += 1000.0 * dt * neurons[k].a *
							(0.2 * neurons[k].v - neurons[k].u)

		# Check if spiked and calculate output
		outval = 0.0
		if neurons[k].v >= 30.0
			spikes = [spikes ; t k] 	# time node
			outval = 1.0				
			neurons[k].v = -65.0
			neurons[k].u += neurons[k].d
		end

		# Update neuron trace
		if k <= n_exc
			spike_post[k] = outval 
			# trace_post[k] = min(1.0, trace_post[k]*(1.0 - (dt/τ_post))+spike_post[k])
			trace_post[k] = trace_post[k]*(1.0 - (dt/τ_post))+spike_post[k]
		end

		# Update synapses
		if k <= n_exc
			for p ∈ 1:length(inputs)
				synapses[k][p] = synapses[k][p] +
								dt * (A_post * trace_pre[k][p] * spike_post[k] -
									  A_pre * trace_post[k] * spike_pre[k][p])
			end
			synapses[k] = clamp.(synapses[k], 0, syn_max)
		end
		spike_post .= 0.0

		# Load the output
		pushoutput!(outval, k, dn)
	end

	# Advance the state
	advance!(dn)
	synapse_w[k] = synapses[1][1]
	syn_trace[k] = trace_pre[1][1]
	neu_trace[k] = trace_post[1]
	potential[k] = neurons[1].v
end

spikes = spikes[2:end,:]
println("Number of random firings: $(rand_count) (expected: $(length(ts)*n/1000))")

end
