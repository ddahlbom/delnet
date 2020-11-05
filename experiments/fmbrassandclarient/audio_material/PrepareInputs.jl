module PrepareInputs

using WAV
using StimAnaGen
using DelNetExperiment
using JLD

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

################################################################################
# Script
################################################################################

## Trim
sigdur = 0.150
decaytime = 0.01
clarsigs  = trimfiles(clarbasename, basepath, freqs, sigdur, decaytime)
brasssigs = trimfiles(brassbasename, basepath, freqs, sigdur, decaytime)

## Normalize
# ref = totalenergy(clarsigs[1])
# clarsigs_normed  = map(x->normalizebyenergy(x,ref), clarsigs)
# brasssigs_normed = map(x->normalizebyenergy(x,ref), brasssigs)
clarsigs_normed = map(x->normalizebymax(x), clarsigs)
brasssigs_normed = map(x->normalizebymax(x), brasssigs)

numchannels = 30
nervesperchannel = 10
refractorytime = 0.00
densityfactor = 1500.0 

clarspikes = [auditoryspikes(c.data, numchannels, nervesperchannel, c.fs;
							 rt=refractorytime, densityfactor=densityfactor)
			  for c ∈ clarsigs]

brassspikes = [auditoryspikes(c.data, numchannels, nervesperchannel, c.fs;
			 				  rt=refractorytime, densityfactor=densityfactor)
			   for c ∈ brasssigs]

end
