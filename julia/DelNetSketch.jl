module DelNetSketch

using Plots

# -------------------- Data Structures --------------------

# mutable struct NodeState
# 
# end

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


# -------------------- Functions --------------------
function advance(input, output, inverseidces, delays, delbuf)	
# function advance(input, inverseidces, delays, delbuf)	
# 	output = zeros(length(input))

	#load input
	for i ∈ 1:length(input)
		buf_idx = delays[i].startidx + delays[i].offset
		delbuf[buf_idx] = input[i]	
	end

	# advance buffer
	for i ∈ 1:length(input)
		delays[i].offset = (delays[i].offset + 1) % delays[i].len
	end
	
	#pull output
	for i ∈ 1:length(input)
		outputs[ inverseidces[i] ] = delbuf[ delays[i].startidx + delays[i].offset ]	
	end
	output
end

function orderbuf(delay, delbuf) 
	[ delbuf[delay.startidx + ((delay.offset + k) % delay.len) ]
	 for k ∈ 0:delay.len-1] |> v -> map(x -> x == 0.0 ? "-" : "$(Int(round(x)))", v) |> reverse |> prod
end

function buftostr(buffer)
	buffer |> v -> map(x -> x == 0.0 ? "-" : "$(Int(round(x)))", v) |> prod
end


function blobnetwork(n, p, delays::Array{Int, 1})
	delmat = rand(n,n) |> m -> map(x -> x < p ? 1 : 0, m)
	for k ∈ 1:n delmat[k,k] = 0 end
	numlines = sum(delmat)
	@assert 0 ∉ delays
	delmat = map( x -> x == 1 ? rand(delays) : 0, delmat)
	return delmat, numlines, sum(delmat)
end

# -------------------- Parameters --------------------
n = 15 		# number of elements
p = 0.1 		
d_max = 3


# ---------- Generate Network, Nodes and Delay Lines --------------------
delmat, numlines, deltot = blobnetwork(n, p, collect(1:d_max))

# for testing
n = 4
delmat = [0 1 1 0; 0 0 1 0; 0 0 0 1; 1 0 0 0 ]
for k ∈ 1:n delmat[k,k] = 0 end
numlines = sum(delmat)
delmat .*= d_max
deltot = sum(delmat)

inputs = zeros(numlines)
outputs = zeros(numlines)
delbuf = zeros(deltot)

nodes = [Node() for _ ∈ 1:n] 
delays = Array{Delay, 1}(undef,numlines)


delcount = 1
startidx = 1
total = 0
for i ∈ 1:n
	global delcount
	global startidx
	global total
	global input_idx
	for j ∈ 1:n
		if delmat[i,j] != 0
			total += delmat[i,j]
			push!(nodes[i].nodes_out, j)
			nodes[i].num_out += 1
			push!(nodes[j].nodes_in, i)
			delays[delcount] = Delay(0, startidx, delmat[i,j], i, j)
			startidx += delmat[i,j]
			delcount += 1
		end
	end
end

# println("Sanity check -- the same?: $(delcount-1), $numlines")
# println("Sanity check -- the same?: $deltot, $total")
num_outputs    = [ nd.num_out for nd ∈ nodes ]
in_base_idcs = [ sum(num_outputs[1:k]) for k ∈ 1:n-1 ]
in_base_idcs = [1; in_base_idcs[1:end] .+ 1]


idx = 1
for i ∈ 1:length(nodes)
	global idx
	nodes[i].num_in = length(nodes[i].nodes_in)
	nodes[i].idx_out_to_in = idx
	idx += nodes[i].num_in
	nodes[i].idx_in_to_out = in_base_idcs[i]
end

num_inputs    = [ nd.num_in for nd ∈ nodes ]
out_base_idcs = [ sum(num_inputs[1:k]) for k ∈ 1:n-1 ]
out_base_idcs = [1; out_base_idcs[1:end] .+ 1]
out_counts = zeros(length(out_base_idcs))

inverseidces = zeros(Int64, length(outputs))
for i ∈ 1:length(inputs)
	inverseidces[i] = out_base_idcs[delays[i].target] +
					  out_counts[delays[i].target]
	out_counts[delays[i].target] += 1
end



num_steps = 6
nodevals = zeros(length(nodes))
nodevals[1] = 1.0
nodevals[2] = 1.0

for j ∈ 1:num_steps
	global inputs, outputs, inverseidces, delays, delbuf, op, nodevals

	for k ∈ 1:length(nodevals)
		# print("Node broadcast $k: ")
		for l ∈ 1:nodes[k].num_out
			inputs[nodes[k].idx_in_to_out+l-1] = nodevals[k]   
			# print(" $(nodes[k].idx_in_to_out+l-1) ($(nodevals[k]))")
		end
		# println()
	end
	# println("Inputs after broadcast: ", inputs)

	println("\nSTEP $j:")
	println("nodevals: $(nodevals)")
	println(inputs |> buftostr, "\n")
	advance(inputs, outputs, inverseidces, delays, delbuf)

	for d ∈ delays
		vals = orderbuf(d, delbuf)
		println("($(d.source)) "*vals*" ($(d.target))")
	end

	# println( delbuf |> buftostr )
	println("\n", outputs |> buftostr, "\n")
	# inputs[:] = outputs[:]
	
	nodevals = zeros(length(nodes))
	for k ∈ 1:length(nodevals)
		# print("Node gather $k: ")
		for l ∈ 1:nodes[k].num_in
			nodevals[k] += outputs[nodes[k].idx_out_to_in+l-1] 
			# print(" $(nodes[k].idx_out_to_in+l-1)")
		end
		nodevals[k] = nodevals[k] > 1.0 ? 1.0 : 0
		# println()
	end
	println("############################################################")
end


end
