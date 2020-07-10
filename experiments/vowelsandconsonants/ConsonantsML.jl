module ConsonantsML

using JLD
using Plots
using DelNetExperiment

DNE = DelNetExperiment

results = load("consonantsresults.jld", "results")

numinstances = 5 
testinstances = [1,2,6,7] 
traininginstances = [1,2,6,7]
trainidcs = findall(x -> x ∈ traininginstances, results.inputids)
testidcs = findall(x -> x ∈ testinstances, results.inputids)
inputids = map(x -> x <= numinstances ? 1 : 2, results.inputids)
rt = results
results = DelNetExperiment.ExperimentOutput(
								 rt.modelname,
								 rt.trialname,
								 rt.tp,
								 rt.input,
								 rt.output,
								 rt.inputtimes,
								 inputids, 	# insert new inputids
								 rt.synapses)

outputneurons = collect(301:800)
inputs = results.input
inputtimes = results.inputtimes
inputids = results.inputids
output = filter(s -> outputneurons[1] <= s.n <= outputneurons[end],
				results.output)




fs = 4000.0
dt = 1.0/fs
maxdelay = 0.02
possibledelays = dt:dt:maxdelay
trialdur = 0.15 	# inputlength plus 0.05 seconds

dl1s = [DNE.DelayLine(rand(possibledelays), []) for _ ∈ outputneurons]
dl2s = [DNE.DelayLine(rand(possibledelays), []) for _ ∈ outputneurons]

d_min = 0.001
d_max = 0.020
weight = 0.25 
μ = 0.005

#### training delays
numtrainingcycles = 5
for i ∈ 1:numtrainingcycles 
	println("Training round $i/$numtrainingcycles")
	for (k,inputtime) ∈ enumerate(inputtimes[trainidcs])
		global correctcount, falsecount, nonanswer
		id = inputids[trainidcs][k]
		spikes = filter(s -> inputtime <= s.t < inputtime + trialdur,
						output)
		neuron1 = DNE.SRMNeuron()
		neuron2 = DNE.SRMNeuron()

		# Load up buffers
		for spike ∈ spikes
			DNE.dlevent(dl1s[spike.n-outputneurons[1]+1], spike.t - inputtime)
			DNE.dlevent(dl2s[spike.n-outputneurons[1]+1], spike.t - inputtime)
		end

		# Run simulation (process events in buffers)
		cat1spikes = []
		cat2spikes = []
		firstspikeinputs1 = []
		firstspikeinputs2 = []
		firstspiked1 = false
		firstspiked2 = false
		for t ∈ 0.0:dt:trialdur+maxdelay+dt
			cat1inputs = []
			cat2inputs = []
			for n ∈ 1:length(outputneurons)
				spiked1 = DNE.dladvance(dl1s[n], dt)
				(spiked1 != 0) && push!(cat1inputs, n)
				spiked2 = DNE.dladvance(dl2s[n], dt)
				(spiked2 != 0) && push!(cat2inputs, n)
			end

			if firstspiked1 == false
				spiked = DNE.srmneuronupdate(neuron1, t, weight*length(cat1inputs))
				if spiked
					firstspikeinputs1 = copy(cat1inputs)
					firstspiked1 = true
					push!(cat1spikes, t)
				end
			end

			if firstspiked2 == false
				spiked = DNE.srmneuronupdate(neuron2, t, weight*length(cat2inputs))
				if spiked
					firstspikeinputs2 = copy(cat2inputs)
					firstspiked2 = true
					push!(cat2spikes, t)
				end
			end

		end

		if length(cat1spikes) == 0 && length(cat2spikes) == 0
			choice = 0
		elseif length(cat1spikes) == 0
			choice = 2
		elseif length(cat2spikes) == 0
			choice = 1
		elseif cat1spikes[1] == cat2spikes[1]
			choice = 0
		elseif cat1spikes[1] < cat2spikes[1]
			choice = 1
		else
			choice = 2
		end

		# stochasti delay line updates
		if length(cat1spikes) > 0 && length(cat2spikes) > 0
			t1 = cat1spikes[1]
			t2 = cat2spikes[1]
			if id == 1
				if t2 - t1 + μ >= 0
					idx1 = rand(firstspikeinputs1)
					idx2 = rand(firstspikeinputs2)
					(dl1s[idx1].dur > d_min) && (dl1s[idx1].dur -= 0.001)
					(dl2s[idx2].dur < d_max) && (dl2s[idx2].dur += 0.001)
				end
			else
				if t1 - t2 + μ >= 0
					idx1 = rand(firstspikeinputs1)
					idx2 = rand(firstspikeinputs2)
					(dl1s[idx1].dur < d_max) && (dl1s[idx1].dur += 0.001)
					(dl2s[idx2].dur > d_min) && (dl2s[idx2].dur -= 0.001)
				end
			end
		end
	end
end

#### Testing
correctcount = 0
falsecount = 0
nonanswer = 0
for (k,inputtime) ∈ enumerate(inputtimes[testidcs])
	global correctcount, falsecount, nonanswer
	id = inputids[testidcs][k]
	spikes = filter(s -> inputtime <= s.t < inputtime + trialdur,
					output)
	neuron1 = DNE.SRMNeuron()
	neuron2 = DNE.SRMNeuron()

	# Load up buffers
	for spike ∈ spikes
		DNE.dlevent(dl1s[spike.n-outputneurons[1]+1], spike.t - inputtime)
		DNE.dlevent(dl2s[spike.n-outputneurons[1]+1], spike.t - inputtime)
	end

	# Run simulation (process events in buffers)
	cat1spikes = []
	cat2spikes = []
	firstspikeinputs1 = []
	firstspikeinputs2 = []
	firstspiked1 = false
	firstspiked2 = false
	for t ∈ 0.0:dt:trialdur+maxdelay+dt
		cat1inputs = []
		cat2inputs = []
		for n ∈ 1:length(outputneurons)
			spiked1 = DNE.dladvance(dl1s[n], dt)
			(spiked1 != 0) && push!(cat1inputs, n)
			spiked2 = DNE.dladvance(dl2s[n], dt)
			(spiked2 != 0) && push!(cat2inputs, n)
		end

		if firstspiked1 == false
			spiked = DNE.srmneuronupdate(neuron1, t, weight*length(cat1inputs))
			if spiked
				firstspikeinputs1 = copy(cat1inputs)
				firstspiked1 = true
				push!(cat1spikes, t)
			end
		end

		if firstspiked2 == false
			spiked = DNE.srmneuronupdate(neuron2, t, weight*length(cat2inputs))
			if spiked
				firstspikeinputs2 = copy(cat2inputs)
				firstspiked2 = true
				push!(cat2spikes, t)
			end
		end

	end

	println("Category 1 spike time: ",
			length(cat1spikes) > 0 ? cat1spikes[1] : "NO SPIKE")
	println("Category 2 spike time: ",
			length(cat2spikes) > 0 ? cat2spikes[1] : "NO SPIKE")

	if length(cat1spikes) == 0 && length(cat2spikes) == 0
		choice = 0
	elseif length(cat1spikes) == 0
		choice = 2
	elseif length(cat2spikes) == 0
		choice = 1
	elseif cat1spikes[1] == cat2spikes[1]
		choice = 0
	elseif cat1spikes[1] < cat2spikes[1]
		choice = 1
	else
		choice = 2
	end

	if choice == id
		correctcount += 1
	elseif choice != id && choice != 0
		falsecount += 1
	else
		nonanswer += 1
	end

	# stochastic delay line updates
	println("Chose: $choice, GT: $id")
	println()
	
end

n = length(inputtimes[testidcs])

println("Correct: $(correctcount/n), False: $(falsecount/n), Nonanswer: $(nonanswer/n)")


end
