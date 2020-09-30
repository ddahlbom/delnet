module ExhaustCombos

struct ThreeGroup
	one
	two
	three
end

function exhaust3naive(vals)
	n = length(vals)
	combos = Array{ThreeGroup,1}(undef,0)
	for i ∈ 1:n
		for j ∈ i+1:n
			for k ∈ j+1:n
				push!(combos, ThreeGroup(i,j,k))
			end
		end
	end
	combos
end

function positionupdate!(positions, which, maxidx)
	numpositions = length(positions)
	done = false
	if which == numpositions
		if positions[which] == maxidx
			positionupdate!(positions, which-1, maxidx)
			positions[which] = positions[which-1]+1
		else
			positions[which] += 1
		end
	elseif which == 1
		if positions[which] == positions[which+1]-1
			return positions
		else
			positions[which] += 1
		end
	else
		if positions[which] == positions[which+1]-1
			positionupdate!(positions, which-1, maxidx)
			positions[which] = positions[which-1]+1
		else
			positions[which] += 1
		end
	end
	return positions	
end

function exhaustn(vals, choosenum)
	positions = collect(1:choosenum)
	combos = Array{Array{Int64,1},1}(undef,0)
	push!(combos, copy(positions))
	n = length(vals)
	@assert n > choosenum "More specified positions than values in list!"
	done = false
	positions_old = zeros(Int64, choosenum) 
	while !done 
		positions_old = copy(positions)
		positionupdate!(positions, choosenum, n)
		if positions == positions_old
			return combos
		else
			push!(combos, copy(positions))
		end
	end
end


end
