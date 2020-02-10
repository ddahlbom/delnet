module DelNet

export blobgraph, delnetfromgraph, pushoutput!,
	   getinputs, advance!, buftostr, orderbuf


# -------------------- Data Structures --------------------
mutable struct Node
	idx_out_to_in::Int64
	num_in::Int64
	nodes_in::Array{Int64,1} 	# just for reference/construction
	idx_in_to_out::Int64
	num_out::Int64
	nodes_out::Array{Int64,1} 	#just for reference/construction
end

Node() = Node(0,0,Int64[],0,0,Int64[])


mutable struct Delay
	offset::Int64
	startidx::Int64
	len::Int64
	source::Int64
	target::Int64
end

mutable struct DelayNetwork
	inputs::Array{Float64,1}
	outputs::Array{Float64,1}
	delaybuf::Array{Float64,1}
	invidx::Array{Int64,1}
	nodes::Array{Node,1}
	delays::Array{Delay,1}
end


# -------------------- Functions --------------------
"""
"""
function pushoutput!(val, idx, dn)
	i1 = dn.nodes[idx].idx_in_to_out
	i2 = i1 + dn.nodes[idx].num_out-1 
	dn.inputs[i1:i2] .= val
end


"""
"""
function getinputs(idx, dn::DelayNetwork)
	i1 = dn.nodes[idx].idx_out_to_in
	i2 = i1 + dn.nodes[idx].num_in - 1
	dn.outputs[i1:i2] 	# note, this copies -- not a reference
end


"""
"""
function advance!(dn::DelayNetwork)
	#load input
	for i ∈ 1:length(dn.inputs)
		dn.delaybuf[ dn.delays[i].startidx + dn.delays[i].offset  ] = dn.inputs[i]	
	end

	# advance buffer
	for i ∈ 1:length(dn.inputs)
		dn.delays[i].offset = (dn.delays[i].offset + 1) % dn.delays[i].len
	end
	
	#pull output
	for i ∈ 1:length(dn.inputs)
		dn.outputs[ dn.invidx[i] ] = dn.delaybuf[ dn.delays[i].startidx + dn.delays[i].offset ]	
	end
	# output
end



"""
"""
function orderbuf(delay, delbuf) 
	[ delbuf[delay.startidx + ((delay.offset + k) % delay.len) ]
	 for k ∈ 0:delay.len-1] |> v -> map(x -> x == 0.0 ? "-" : "$(Int(round(x)))", v) |> reverse |> prod
end


"""
"""
function buftostr(buffer)
	buffer |> v -> map(x -> x == 0.0 ? "-" : "$(Int(round(x)))", v) |> prod
end


"""
"""
function blobgraph(n, p, delays::Array{Int, 1})
	@assert 0 ∉ delays "1 corresponds to no delay -- can't have 0 in delays"
	delmat = rand(n,n) |> m -> map(x -> x < p ? rand(delays) : 0, m)
	for k ∈ 1:n delmat[k,k] = 0 end
	return delmat
end


"""
"""
function delnetfromgraph(graph)
	n = size(graph)[1]
	deltot = sum(graph) 	# total number of delays to allocate
	numlines = sum( map(x -> x != 0 ? 1 : 0, graph) )
	inputs = zeros(numlines)
	outputs = zeros(numlines)
	delbuf = zeros(deltot)
	nodes_in = [[] for _ ∈ 1:n]

	nodes = [Node() for _ ∈ 1:n] 
	delays = Array{Delay, 1}(undef,numlines)


	delcount = 1
	startidx = 1
	total = 0
	for i ∈ 1:n
		for j ∈ 1:n
			if graph[i,j] != 0
				total += graph[i,j]
				nodes[i].num_out += 1
				push!(nodes_in[j], i)
				delays[delcount] = Delay(0, startidx, graph[i,j], i, j)
				startidx += graph[i,j]
				delcount += 1
			end
		end
	end

	num_outputs  = [ nd.num_out for nd ∈ nodes ]
	in_base_idcs = [ sum(num_outputs[1:k]) for k ∈ 1:n-1 ]
	in_base_idcs = [ 1; in_base_idcs[1:end] .+ 1 ]

	idx = 1
	for i ∈ 1:length(nodes)
		nodes[i].num_in = length( nodes_in[i] )
		nodes[i].idx_out_to_in = idx
		idx += nodes[i].num_in
		nodes[i].idx_in_to_out = in_base_idcs[i]
	end

	num_inputs    = [ nd.num_in for nd ∈ nodes ]
	out_base_idcs = [ sum(num_inputs[1:k]) for k ∈ 1:n-1 ]
	out_base_idcs = [ 1; out_base_idcs[1:end] .+ 1 ]
	out_counts = zeros(length(out_base_idcs))

	inverseidces = zeros(Int64, length(outputs))
	for i ∈ 1:length(inputs)
		inverseidces[i] = out_base_idcs[delays[i].target] + out_counts[delays[i].target]
		out_counts[delays[i].target] += 1
	end
		
	DelayNetwork(inputs, outputs, delbuf, inverseidces, nodes, delays)
end

end
