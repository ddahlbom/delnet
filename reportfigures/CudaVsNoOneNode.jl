module CudaVsNoOneNode

using Plots
using Formatting

numnodes = [800, 1131, 1600, 2263, 3200, 4525]
numsyn   = [64000, 128000, 256000, 512000, 1024000, 4096000]

totals_cu = [1075.963125, 2341.053178, 4465.354660, 8838.505402, 19679.607812, 38804.135236] ./ 1000.0
t_inputs_cu = [0.104213, 0.217258, 0.445819, 0.878838, 1.779765, 3.553215]
t_syntraces_cu = [0.240776, 0.285203, 0.308995, 0.341920, 0.394423, 0.416434]
t_neurons_cu = [0.012951, 0.018114, 0.025937, 0.036299, 0.051531, 0.072856]
t_spiked_cu = [0.000885, 0.001341, 0.007897, 0.014552, 1.864528, 3.385552]
t_pushbuf_cu = [0.049622, 0.081276, 0.145494, 0.286411, 0.596657, 1.182108]
t_neutracs_cu = [0.012951, 0.018114, 0.025937, 0.036299, 0.051531, 0.072856]
t_syns_cu = [0.222800, 0.392573, 0.704250, 1.281902, 2.615458, 5.090564]
t_advbuf_cu = [0.443649, 1.343769, 2.825131, 5.939421, 12.302679, 25.022309]

totals_nc = [965.666750, 2402.465491, 6403.812858, 12773.922443, 27989.524493, 54604.265128] ./ 1000.0
t_inputs_nc = [0.104299, 0.216253, 0.431653, 0.875638, 1.776694, 3.553212]
t_syntraces_nc = [0.224185, 0.465278, 2.289251, 4.210844, 9.306203, 18.549352]
t_neurons_nc = [0.012881, 0.018174, 0.025817, 0.036633, 0.051760, 0.072932]
t_spiked_nc = [0.000933, 0.001395, 0.009299, 0.014642, 1.195865, 0.348494]
t_pushbuf_nc = [0.042808, 0.074596, 0.135739, 0.289774, 0.636160, 1.203149]
t_neutraces_nc = [0.012881, 0.018174, 0.025817, 0.036633, 0.051760, 0.072932]
t_syns_nc = [0.171770, 0.318452, 0.615326, 1.227327, 2.569688, 5.121966 ]
t_advbuf_nc = [0.407702, 1.306762, 2.894587, 6.115827, 12.448801, 25.749351]



timetickvals = collect(2 .^ (range(0.0, 5.0, length=6)))
timeticklabels = [format("{:.1f}", x) for x ∈ timetickvals]
# timetickvals = collect(range(0.0, 60.0, length=5))


p1 = plot(xlabel="Mean Number of Synapses",
		  ylabel="Execution Time (s)",
		  lw=2.0,
		  yticks=(timetickvals, timeticklabels),
		  xtickfontsize=10,
		  ytickfontsize=10,
		  guidefontsize=11,
		  xscale=:log2,
		  yscale=:log2)
plot!(p1, numsyn, totals_cu,
	  label="CUDA",
	  lw=2.0,
	  markershape=:circle,
	  markersize=5.0)
plot!(p1, numsyn, totals_nc,
	  label="No CUDA",
	  lw=2.0,
	  markershape=:circle,
	  markersize=5.0)


end
