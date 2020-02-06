module DelNetSketch

using Plots

mutable struct Node
	# the lists should just be "pointers" -- but try once with eliminating
	# them to see if it enhances performance
	startidx_in::Int64
	num_in::Int64
	nodes_in::Array{Int64,1} 	# just for reference/construction
	startidx_out::Int64
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
n = 100 		# number of elements
p = 0.1 		
d_max = 20 		


# Generate connectivity matrix
# delmat = rand(n,n) |> m -> map(x -> x < p ? rand(0:d_max) : 0, m)
delmat = rand(n,n) |> m -> map(x -> x < p ? 1 : 0,m)
numlines = sum(delmat)
delmat = map(x -> x == 1 ? rand(1:d_max) : 0, delmat)
deltot = sum(delmat)

inputs = zeros(numlines)
outputs = zeros(numlines)
delbuf = zeros(deltot)

nodes = [Node() for _ ∈ 1:n] 
delays = Array{Delay, 1}(undef,numlines)

e_forw = []
e_revr = []


delcount = 1
startidx = 1
input_idx = 1
output_idx = 1
total = 0
for i ∈ 1:n
	global delcount
	global startidx
	global total
	global input_idx
	for j ∈ 1:n
		if delmat[i,j] != 0
			println("Delay count: $delcount")
			total += delmat[i,j]
			push!(e_forw, (i,j))				
			push!(e_revr, (j,i))				
			push!(nodes[i].nodes_out, j)
			nodes[i].num_out += 1
			push!(nodes[j].nodes_in, i)
			nodes[j].num_in += 1
			nodes[i].startidx_in = input_idx
			delays[delcount] = Delay(0, startidx, delmat[i,j], i, j)
			startidx += delmat[i,j]
			delcount += 1
			input_idx += 1
		end
	end
end

edges = sort(e_revr)

function delnet_advance(input, output, edges, delays, delbuf)	
	for i ∈ 1:length(input)
		buf_idx = delays[i].startidx + delays[i].offset
		# load input
		delbuf[buf_idx] = input[i]	
		# pull output
		output[edges[i][2]] = delbuf[buf_idx]
	end
	
	# advance buffer
	for i ∈ 1:length(input)
		delays[i].offset = (delays[i].offset + 1) % delays[i].length + 1
	end
end



end
