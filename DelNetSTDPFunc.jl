module DelNetSTDPFunc

using Plots
using Random: randperm
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

struct SimParams
	a_exc::Float64
	a_inh::Float64
	d_exc::Float64
	d_inh::Float64
	w_exc::Float64
	w_inh::Float64
	τ_pre::Float64
	A_pre::Float64
	τ_post::Float64
	A_post::Float64
	syn_max::Float64
	n_exc::Int64
	n_inh::Int64
	p_contact::Float64
	d_max::Int64
end

SimParams() = SimParams(0.02, 	#a_exc
					    0.1, 	#a_inh
					    8.0, 	#d_exc
					    2.0, 	#d_inh
					    6.0, 	#w_exc
					    -5.0, 	#w_inh
					    0.02, 	#τ_post
					    0.12, 	#A_pre
					    0.02, 	#τ_pre
					    0.1, 	#A_post
					    10.0, 	#syn_max
						800, 	#n_exc
						200, 	#n_inh
						0.1, 	#p_contact
						20) 	#d_max

# ---------- Generate Network, Nodes and Delay Lines --------------------
trialparams = SimParams()


# -------------------- Run Simulation --------------------
function runsim(params, fs, dur; blocksize=:none, outfile="trialdata")
	n = params.n_exc + params.n_inh 		# number of elements
	#spikes = [0.0 0.0]
	f = open(outfile, "w")
	if blocksize == :none
		blocksize = 10*n
	end
	spikes = zeros(blocksize, 2)
	spike_count = 0
	spike_count_offset = 0
	rand_count = 0

	# Generate graph
	# g = blobgraph(n, params.p_contact, collect(2:params.d_max+1))
	# g[n_exc+1:end,:] = map(x -> x != 0 ? 1 : 0, g[n_exc+1:end,:]) # inh delays = 1
	
	g = zeros(Int64,n,n)
	for k ∈ 1:params.n_exc
		g[k, randperm(n)[1:Int(round(params.p_contact*n))]] =
		rand(2:params.d_max+1,Int(round(params.p_contact*n)))
	end
	g[params.n_exc+1:end,:] .= 0
	for k ∈ (params.n_exc+1):n
		g[k, randperm(params.n_exc)[1:Int(round(params.p_contact*n))]] .= 2
	end
	# g[1:params.n_exc, params.n_exc+1:end] = map( x -> x != 0 ? 2 : 0, g[1:params.n_exc, params.n_exc+1:end])

	dn = delnetfromgraph(g)

	invinvidx = Array{Int64,1}(undef, length(dn.invidx))
	for k ∈ 1:length(invinvidx)
		invinvidx[dn.invidx[k]] = k
	end

	# Initialize neurons
	neurons    = [Neuron() for _ ∈ 1:n]
	synapses   = [Float64[] for _ ∈ 1:n]
	trace_pre  = [Float64[] for _ ∈ 1:params.n_exc]
	spike_pre  = [Float64[] for _ ∈ 1:params.n_exc]
	trace_post = zeros(params.n_exc)
	spike_post = zeros(params.n_exc)

	for i ∈ 1:params.n_exc
		neurons[i].a = params.a_exc
		neurons[i].d = params.d_exc 
		synapses[i]  = params.w_exc .* ones(dn.nodes[i].num_in)
		trace_pre[i] = zeros(dn.nodes[i].num_in)
		spike_pre[i] = zeros(dn.nodes[i].num_in)
	end

	for i ∈ params.n_exc+1:n
		neurons[i].a = params.a_inh
		neurons[i].d = params.d_inh
		synapses[i]  = params.w_inh .* ones(dn.nodes[i].num_in)
	end

	dt = 1/fs
	ts = collect(0:dt:dur)
	for (k, t) ∈ enumerate(ts)
		( (k%100) == 0 ) && ( println("Time: $(t+dt)") )
		# update node and calculate ouput, push into delnet
		for k ∈ 1:n

			# Get input from delay lines
			inputs = getinputs(k,dn)
			inval = sum(synapses[k] .* inputs)

			# Update synapse traces
			if k <= params.n_exc
				for p ∈ 1:length(inputs)
					spike_pre[k][p] = inputs[p]
					#trace_pre[k][p] = min(1.0, trace_pre[k][p]*(1.0-(dt/τ_pre))+spike_pre[k][p])
					trace_pre[k][p] = trace_pre[k][p]*(1.0-(dt/params.τ_pre))+spike_pre[k][p]
				end
			end

			# Random input (noise)
			if rand() < 1.0/n
				inval += 20.0 * (fs/1000.0)
				rand_count += 1
			end

			# Update neuron state
			neurons[k].v += 500.0 * dt .* ((0.04 * neurons[k].v + 5.0) * neurons[k].v +
									140.0 - neurons[k].u + inval)
			neurons[k].v += 500.0 * dt .* ((0.04 * neurons[k].v + 5.0) * neurons[k].v +
									140.0 - neurons[k].u + inval)
			neurons[k].u += 1000.0 * dt * neurons[k].a *
								(0.2 * neurons[k].v - neurons[k].u)

			# Check if spiked and calculate output
			outval = 0.0
			if neurons[k].v >= 30.0
				#spikes = [spikes ; t k] 	# time node
				spike_count += 1
				spikes[spike_count,:] = [t k]
				outval = 1.0				
				neurons[k].v = -65.0
				neurons[k].u += neurons[k].d
			end

			# Update neuron trace
			if k <= params.n_exc
				spike_post[k] = outval 
				# trace_post[k] = min(1.0, trace_post[k]*(1.0 - (dt/τ_post))+spike_post[k])
				trace_post[k] = trace_post[k]*(1.0 - (dt/params.τ_post))+spike_post[k]
			end

			# Update synapses
			if k <= params.n_exc
				for p ∈ 1:length(inputs)
					synapses[k][p] = synapses[k][p] + 0.00001 +
									dt * (params.A_post * trace_pre[k][p] * spike_post[k] -
										  params.A_pre * trace_post[k] * spike_pre[k][p])
				end
				synapses[k] = clamp.(synapses[k], 0, params.syn_max)
			end
			spike_post .= 0.0

			# Load the output
			pushoutput!(outval, k, dn)
		end

		# Advance the state
		advance!(dn)

		# Save spikes if necessary
		if spike_count > blocksize - n
			for l ∈ 1:spike_count
				write(f, "$(spikes[l,1])  $(Int(round(spikes[l,2]))) \n")
			end
			spikes .= 0
			spike_count = 0
		end

	end

	for l ∈ 1:spike_count
		write(f, "$(spikes[l,1])  $(Int(round(spikes[l,2]))) \n")
	end
	close(f)

	println("Number of random firings: $(rand_count) (expected: $(length(ts)*n/1000))")
	return g, synapses
end


end
