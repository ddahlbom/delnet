module CorGen

export uniformblock, analyzeblock

using Plots; pyplot()

dist( p1, p2 ) = sqrt(sum((p1 .- p2) .^ 2))

"""
x, y, z in mm.  ρ in #/mm^3.  λ tbd. v ∈ mm/s
"""
function uniformblock(x, y, z, ρ, λ, v; verbose=false, maxlen=1.0)
	n = Int(round(x * y * z * ρ))
	delmat = zeros(n,n)
	ps = [(rand()*x, rand()*y, rand()*z) for _ ∈ 1:n]
	λsq = λ^2

	# Determine connections and delays
	for i ∈ 1:n
		for j ∈ 1:n
			d = dist( ps[i], ps[j] )
			p_contact = λsq * d * exp(-λ*d)
			if (d < maxlen) && (rand() < p_contact)
				delmat[i,j] = (d*(1.1+rand())) / v
			end
		end
	end
		
	if verbose
		analyzeblock(delmat)
	end

	return ps, delmat
end

function analyzeblock(delmat::Array{Float64,2})
	n = size(delmat)[1]
	conmat = map(x -> x != 0.0 ? 1 : 0, delmat)
	numdels = sum(conmat)
	println("Number of nodes:\t$(n)")
	println("Probability of contact:\t$(sum(conmat)/(n*n))")
	println("Average delay:\t$(sum(1000.0 .* delmat)/numdels) ms")
end

function delayhistvals(delmat::Array{Float64,2})
	n = size(delmat)[1]
	numdels = sum( map( x -> x != 0.0 ? 1 : 0, delmat ) )
	vals = zeros(numdels)
	idx = 1
	for i ∈ 1:n
		for j ∈ 1:n
			if delmat[i,j] != 0.0
				vals[idx] = delmat[i,j] 
				idx += 1
			end
		end
	end
	vals
end

function plotnodenet(node, ps, delmat)
	idcs = findall(x -> x != 0.0, delmat[node,:])	
	idcs = [node; idcs]

	plt = scatter( [ps[k][1] for k ∈ setdiff(1:length(ps),idcs)],
				   [ps[k][2] for k ∈ setdiff(1:length(ps),idcs)],
				   [ps[k][3] for k ∈ setdiff(1:length(ps),idcs)],
				   markeralpha=0.1,
				   markerstrokealpha=0.10,
				   markerstrokewidth=0.2)

	scatter!(plt, [ps[k][1] for k ∈ idcs],
			 	  [ps[k][2] for k ∈ idcs],
				  [ps[k][3] for k ∈ idcs],
				  lc=:red )

	return plt
end


end
