module CorGen

using Plots; pyplot()

dist( p1, p2 ) = sqrt(sum((p1 .- p2) .^ 2))

"""
x, y, z in mm.  ρ in n/mm.  λ tbd. v ∈ mm/s
"""
function uniformblock(x, y, z, ρ, λ, v)
	n = Int(round(x * y * z * ρ))
	delmat = zeros(n,n)
	ps = [(rand()*x, rand()*y, rand()*z) for _ ∈ 1:n]
	λsq = λ^2

	# Determine connections and delays
	for i ∈ 1:n
		for j ∈ 1:n
			d = dist( ps[i], ps[j] )
			p_contact = λsq * d * exp(-λ*d)
			println(p_contact)
			if rand() < p_contact
				delmat[i,j] = d / v
			end
		end
	end
		

	return ps, delmat
end

	


end
