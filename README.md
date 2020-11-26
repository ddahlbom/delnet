# delnet

C/MPI-based simulation framework for spiking point neurons with conduction delays and STDP. The framework
permits general models with Izhikevich neurons connected together with conduction delays. It also implements
spike-timing dependent plasticity for synapse strengths. This permits simulation of networks of the type
described by Izhikevich (2006) (code [here](http://www.izhikevich.org/publications/spnet.htm)), but it is a much more general framework supporting arbitrary network
topologies.  It requires a series of binary configuration files to run.  These can be set up with the
[companion framework](https://github.com/analogouscircuit/DelNetExperiment) written in Julia.  This framework is specifically designed to be run on a distributed
(MPI) system.


![column activity](/images/testanimation-2000-30-2.gif)

This is under active development for research purposes.  Provided only for the curious, without a warranty
of any kind.


## References
Izhikevich, E. M. (2006). "Polychronization: Computation with Spikes," Neural Computational 18, 245-282.
