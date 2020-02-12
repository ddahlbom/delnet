module DelNetIzhikevich

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

w_exc = 6.0
w_inh = -5.0


# -------------------- Parameters --------------------
fs = 1000.0
dt = 1/fs
dur = 10.0
ts = collect(0:dt:dur)
n_exc = 800
n_inh = 200
n = n_exc + n_inh 		# number of elements
p = 0.1 		
d_max = 20
verbose = false
noise = 0.0 	# probability of random firing
num_steps = 1000

# ---------- Generate Network, Nodes and Delay Lines --------------------

# Generate graph
g = blobgraph(n, p, collect(1:d_max))
g[n_exc+1:end,:] = map(x -> x != 0 ? 1 : 0, g[n_exc+1:end,:])
dn = delnetfromgraph(g)

# Initialize neurons
neurons = [Neuron() for _ ∈ 1:n]
for i ∈ 1:n_exc
	neurons[i].a = a_exc
	neurons[i].d = d_exc 
end

for i ∈ n_exc+1:n
	neurons[i].a = a_inh
	neurons[i].d = d_inh
end


# -------------------- Run Simulation --------------------
spikes = [0.0 0.0]
for t ∈ ts 
	global dn, spikes

	# update node and calculate ouput, push into delnet
	for k ∈ 1:n
		# Get input from delay lines
		w = n <= n_exc ? w_exc : w_inh
		inval = sum( getinputs(k, dn) .* w )
		if rand() < 1.0/n
			inval += 20.0
		end

		# Update state
		neurons[k].v += 500.0 * dt .* ((0.04 * neurons[k].v + 5.0) * neurons[k].v
								+ 140.0 - neurons[k].u + inval)
		neurons[k].v += 500.0 * dt .* ((0.04 * neurons[k].v + 5.0) * neurons[k].v
								+ 140.0 - neurons[k].u + inval)
		neurons[k].u += 1000.0 * dt * neurons[k].a * (0.2 * neurons[k].v - neurons[k].u)

		# Check if spikes and calculate output
		outval = 0.0
		if neurons[k].v >= 30.0
			spikes = [spikes ; t k] 	# time node
			outval = 1.0				
			neurons[k].v = -65.0
			neurons[k].u += neurons[k].d
		end

		# Load the output
		pushoutput!(outval, k, dn)
	end

	# Advance the state
	advance!(dn)
end

spikes = spikes[2:end,:]

end
