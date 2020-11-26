module PrepareInputs

using WAV
using StimAnaGen
using DelNetExperiment
using JLD
using Plots

basepath = "/home/dahlbom/research/delnet/experiments/fmbrassandclarient/audio_material/"


clarbasename = "FMClar"
brassbasename = "FMbrass"

freqs = ["220", "233-1", "246-9", "261-6", "277-2", "293-7",
		 "311-1", "329-6", "349-2", "370-0", "392-0", "415-3"]


function trimfiles(basename, basepath, freqs, duration, decaytime; threshold=0.001)
	trimmed = Array{Signal,1}(undef,0)
	for (k,freq) ∈ enumerate(freqs)
		sig = loadsignal(basepath*basename*freq*".wav")
		sig_trimmed = trimsound(sig, duration, decaytime; threshold=threshold)
		# wavwrite(sig_trimmed.data, 
		# 		 basepath*basename*"_$k"*".wav",
		# 		 Fs=sig_trimmed.fs)
		push!(trimmed, sig_trimmed)
	end
	trimmed
end

export totalenergy, normalize 

totalenergy(s::Signal) = s.data |> x -> x.^2 |> sum   

function normalizebyenergy(s::Signal, ref=1.0)
	Signal(s.data./sqrt(totalenergy(s)) |> x -> x.*sqrt(ref),
		   s.fs)
end

function normalizebymax(s::Signal, ref=1.0)
	Signal(s.data./maximum(s.data) |> x -> x.*ref,
		   s.fs)
end


function lpf(s; initval=1.0)
	outdata = zeros(Float64,length(s))
	outdata[1] = initval
	for i ∈ 2:length(s)
		outdata[i] = 0.5*(s[i]+outdata[i-1])
	end
	outdata
end


function finishedac(spikes::Array{Spike,1}, fs=44100.0, maxdelay=0.01)
	ts, ac = DelNetExperiment.denseac(spikes, fs)
	ac = ac |> lpf |> lpf |> lpf |> lpf
	is = findall(x->0<=x<maxdelay, ts)
	return ts[is], ac[is]
end

################################################################################
# Script
################################################################################

## Trim
sigdur = 0.175
decaytime = 0.01
clarsigs  = trimfiles(clarbasename, basepath, freqs, sigdur, decaytime)
brasssigs = trimfiles(brassbasename, basepath, freqs, sigdur, decaytime)

## Normalize
# ref = totalenergy(clarsigs[1])
# clarsigs_normed  = map(x->normalizebyenergy(x,ref), clarsigs)
# brasssigs_normed = map(x->normalizebyenergy(x,ref), brasssigs)
clarsigs_normed = map(x->normalizebymax(x), clarsigs)
brasssigs_normed = map(x->normalizebymax(x), brasssigs)

numchannels = 20
nervesperchannel = 20
refractorytime = 0.01
densityfactor = 5000.0 

clarspikes = [auditoryspikes(c.data, numchannels, nervesperchannel, c.fs;
							 rt=refractorytime, densityfactor=densityfactor)
			  for c ∈ clarsigs]
brassspikes = [auditoryspikes(c.data, numchannels, nervesperchannel, c.fs;
			 				  rt=refractorytime, densityfactor=densityfactor)
			   for c ∈ brasssigs]


################################################################################
# Analysis
################################################################################
clarts, clarac   = DelNetExperiment.denseac(clarspikes[1], 44100.0)
brassts, brassac = DelNetExperiment.denseac(brassspikes[1], 44100.0)
clarac_smoothed  = clarac  |> lpf |> lpf |> lpf |> lpf # |> lpf |> lpf |> lpf |> lpf
brassac_smoothed = brassac |> lpf |> lpf |> lpf |> lpf # |> lpf |> lpf |> lpf |> lpf
claridcs  = findall(x -> 0 <= x < 0.01, clarts)
brassidcs = findall(x -> 0 <= x < 0.01, brassts)
p_spikeac = plot(clarts[claridcs], clarac_smoothed[claridcs])
plot!(p_spikeac, brassts[brassidcs], brassac_smoothed[brassidcs])

clarsigac, clarsigts = fftac(clarsigs[1]; maxdelay=0.01)
brasssigac, brasssigts = fftac(brasssigs[1]; maxdelay=0.01)
clarsigidcs  = findall(x -> 0 <= x < 0.01, clarsigts)
brasssigidcs = findall(x -> 0 <= x < 0.01, brasssigts)
p_sigac = plot(clarsigts[clarsigidcs], clarsigac[clarsigidcs])
plot!(p_sigac, brasssigts[brasssigidcs], brasssigac[brasssigidcs])

l = @layout [a; b]
p_ac = plot(p_sigac, p_spikeac, layout=l)

p_clarspike = spikeraster(clarspikes[1], ylims=(0,numchannels*nervesperchannel))
p_brassspike = spikeraster(brassspikes[1], ylims=(0,numchannels*nervesperchannel))
p_spikes = plot(p_clarspike, p_brassspike)

println("Average clarinet firing rate: ",
		length(clarspikes[1])/(numchannels*nervesperchannel*sigdur))

save("/home/dahlbom/research/delnet/experiments/fmbrassandclarient/inputspikes.jld",
	 "clarsigs", clarsigs,
	 "brasssigs", brasssigs,
	 "clarspikes", clarspikes,
	 "brassspikes", brassspikes,
	 "numchannels", numchannels,
	 "nervesperchannel", nervesperchannel)

end
