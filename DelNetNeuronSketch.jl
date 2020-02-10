module DelNetNeuronSketch

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
end

a_exc = 0.02
a_inh = 0.1
d_exc = 8.0
d_inh = 2.0

w_exc = 6.0
w_inh = -5.0


# -------------------- Parameters --------------------
n = 1000 		# number of elements
p = 0.1 		
d_max = 20
graph = blobgraph(n, p, collect(1:d_max))


# ---------- Generate Network, Nodes and Delay Lines --------------------
verbose = true
noise = 0.0 	# probability of random firing
num_steps = 10


# Graph generates delay network
# graph_test = [0 0 4 4 0 0; 0 0 0 4 0 0; 0 0 0 0 4 0; 0 4 0 0 0 0; 0 0 0 0 0 0; 0 0 0 0 0 0]
graph_test = 4 .* [0 1 1 0 0; 0 0 1 0 0; 0 0 0 1 0; 1 0 0 0 0; 0 0 0 0 0]
graph = graph_test
dn = delnetfromgraph(graph)

# simple state for each node -- initial conditions
nodevals = zeros( length(dn.nodes) )
nodevals[1] = 1.0
nodevals[2] = 1.0


# -------------------- Run Simulation --------------------
for j ∈ 1:num_steps
	global dn, nodevals

	# Push in node values
	for k ∈ 1:length(nodevals)
		pushoutput!(nodevals[k], k, dn)
	end

	# Advance the state
	advance!(dn)

	# Pull out delay output and use as node input, update node state
	nodevals .*= 0
	for k ∈ 1:length(nodevals)
		nodevals[k] = sum( getinputs(k, dn) )
		nodevals[k] = nodevals[k] > 1.0 ? 1.0 : 0 # only fire if coincidence
	end
	

	# Print output if desired
	if verbose
		println("\nSTEP $j:")
		println("nodevals: $(nodevals)")
		println(dn.inputs |> buftostr, "\n")
		for d ∈ dn.delays
			vals = orderbuf(d, dn.delaybuf)
			println("($(d.source)) "*vals*" ($(d.target))")
		end
		println("\n", dn.outputs |> buftostr, "\n")
		println("############################################################")
	end
end


end
