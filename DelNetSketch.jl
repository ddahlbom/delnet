module DelNetSketch

using Plots

mutable struct Node
	# the lists should just be "pointers" -- but try once with eliminating
	# them to see if it enhances performance
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

# -------------------- Parameters --------------------
n = 25 		# number of elements
p = 0.1 		
d_max = 10


# Generate connectivity matrix
delmat = rand(n,n) |> m -> map(x -> x < p ? 1 : 0,m)
for k ∈ 1:n delmat[k,k] = 0 end
numlines = sum(delmat)
delmat = map(x -> x == 1 ? rand(1:d_max) : 0, delmat)
deltot = sum(delmat)

# for testing
# delmat = [0 1 1 0; 0 0 1 0; 0 0 0 1; 1 0 0 0 ]
# for k ∈ 1:n delmat[k,k] = 0 end
# numlines = sum(delmat)
# delmat .*= d_max
# deltot = sum(delmat)

inputs = zeros(numlines)
outputs = zeros(numlines)
delbuf = zeros(deltot)

nodes = [Node() for _ ∈ 1:n] 
delays = Array{Delay, 1}(undef,numlines)

# e_forw = []
# e_revr = []


delcount = 1
startidx = 1
output_idx = 1
total = 0
for i ∈ 1:n
	global delcount
	global startidx
	global total
	global input_idx
	for j ∈ 1:n
		if delmat[i,j] != 0
			total += delmat[i,j]
			# push!(e_forw, (i,j))				
			# push!(e_revr, (j,i))				
			push!(nodes[i].nodes_out, j)
			nodes[i].num_out += 1
			push!(nodes[j].nodes_in, i)
			nodes[j].num_in += 1
			delays[delcount] = Delay(0, startidx, delmat[i,j], i, j)
			startidx += delmat[i,j]
			delcount += 1
		end
	end
end


idx = 1
for i ∈ 1:length(nodes)
	global idx
	nodes[i].num_in = length(nodes[i].nodes_in)
	nodes[i].idx_out_to_in = idx
	idx += nodes[i].num_in
end

#println("Sanity check -- the same?: $deltot, $total")
num_inputs    = [ nd.num_in for nd ∈ nodes ]
out_base_idcs = [ sum(num_inputs[1:k]) for k ∈ 1:n-1 ]
out_base_idcs = [1; out_base_idcs[1:end] .+ 1]
out_counts = zeros(length(out_base_idcs))


num_outputs    = [ nd.num_out for nd ∈ nodes ]
in_base_idcs = [ sum(num_outputs[1:k]) for k ∈ 1:n-1 ]
in_base_idcs = [1; in_base_idcs[1:end] .+ 1]

for i ∈ 1:length(nodes)
	nodes[i].idx_in_to_out = in_base_idcs[i]
	#nodes[i].num_out = num_inputs[i]
end

inverseidces = zeros(Int64, length(outputs))
for i ∈ 1:length(inputs)
	#println(i)
	inverseidces[i] = out_base_idcs[delays[i].target] + out_counts[delays[i].target]
	out_counts[delays[i].target] += 1
end




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

# outputs[1] = 1.0
#delbuf[1] = 1.0
num_steps = 15 
nodevals = zeros(length(nodes))
nodevals[1] = 1.0
nodevals[2] = 1.0
op = (+)
for j ∈ 1:num_steps
	global inputs, outputs, inverseidces, delays, delbuf, op, nodevals
	# for (k,nd) ∈ enumerate(nodes)
	# 	println("From indices: $(nd.idx_out_to_in) to $(nd.idx_out_to_in + nd.num_in - 1)")
	# 	invals = outputs[ nd.idx_out_to_in : nd.idx_out_to_in + nd.num_in - 1 ]
	# 	println("Invals: $invals")
	# 	val = sum(invals)
	# 	println("To indices: $(nd.idx_in_to_out) to $(nd.idx_in_to_out + nd.num_out - 1)")
	# 	inputs[ nd.idx_in_to_out: nd.idx_in_to_out + nd.num_out - 1 ] .= val
	# end
	
	# println("Inputs before broadcast: ", inputs)
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
	# println("\n", outputs |> buftostr, "\n")
	# inputs[:] = outputs[:]
	
	nodevals = zeros(length(nodes))
	for k ∈ 1:length(nodevals)
		# print("Node gather $k: ")
		for l ∈ 1:nodes[k].num_in
			nodevals[k] += outputs[nodes[k].idx_out_to_in+l-1] 
			# print(" $(nodes[k].idx_out_to_in+l-1)")
		end
		# println()
	end
	println("############################################################")
end


end
