module STDPTest

using Plots
using Random, Distributions

################################################################################
# Functions
################################################################################

function poissonproc(λ, dur)
	times = []
	time = 0.0
	while time < dur
		interval = - log( 1.0 - rand() ) / λ
		time += interval
		if time <= dur
			push!(times, time)
		end
	end
	times
end

mutable struct synapse
	x::Float64
end

################################################################################
# Script
################################################################################

fs = 50e3
dt = 1.0/fs
dur = 1.0
ts = collect(0:dt:dur-dt)
λ = 20.0 	# 10 Hz 

w = zeros(length(ts))

x_pre  = zeros(length(ts))
x_post = zeros(length(ts))

spikes_pre  = zeros(length(ts)) 
spikes_post = zeros(length(ts))

times = poissonproc(λ, dur)
# τ = 1/λ
# times = [τ * k for k ∈ 1:(dur/τ - τ)]
idcs = Int.(round.(times*fs))
idcs = filter(x -> x <= length(ts), idcs)
spikes_pre[idcs] .= 1.0
#times = poissonproc(λ, dur)
times .+= 0.0025
idcs = Int.(round.(times*fs))
idcs = filter(x -> x <= length(ts), idcs)
spikes_post[idcs] .= 1.0

p_spikes_1 = plot(ts, spikes_pre)
p_spikes_2 = plot(ts, spikes_post)
p_spikes   = plot(p_spikes_1, p_spikes_2, layout=(2,1))

x_pre = zeros(length(ts))
τ_pre = 0.034
A_pre = 51.0

x_post = zeros(length(ts))
τ_post = 0.014
A_post = 103.0

s = zeros(length(ts)); s[1] = 6.0

for k ∈ 1:length(x_pre)-1 
	if spikes_post[k] == 1.0
		println("Prespike. Contribution: $(-A_pre * x_pre[k] * spikes_post[k])")
	end
	if spikes_pre[k] == 1.0
		println("Postspike. Contribution: $(A_post * x_post[k] * spikes_pre[k])")
	end
	x_pre[k+1]  = x_pre[k] - (dt/τ_pre) * x_pre[k] + spikes_pre[k]
	x_post[k+1] = x_post[k] - (dt/τ_post) * x_post[k] + spikes_post[k]
	s[k+1] = s[k] + dt * (  A_pre  * x_pre[k]  * spikes_post[k] -
						     A_post * x_post[k] * spikes_pre[k] )
end

p_1 = plot(ts, x_pre)
p_2 = plot(ts, x_post)

p_syns = plot(p_1, p_2, layout=(2,1))

p = plot(p_spikes, p_syns, layout=(1,2))
p2 = plot(ts, s)


end
