module NeuronNumerical

using Plots

v_init = -65.0
u_init = -13.0
a_exc = 0.02
d_exc = 8.0
a_inh = 0.1
d_inh = 2.0

################################################################################
# Functions 
################################################################################

# -------------------- as in C simulation --------------------
f1(v, u, input) = (0.04*v + 5.0)*v + 140.0 - u + input
f2(v, u, a) = a*(0.2*v - u)

function neuronupdate_rk4(v, u, a, input, h) 
	half_h = 0.5*h
	sixth_h = h/6.0

	K1 = f1(v, u, input)
	L1 = f2(v, u, a)

	K2 = f1(v + half_h*K1, u + half_h*L1, input)
	L2 = f2(v + half_h*K1, u + half_h*L1, a)

	K3 = f1(v + half_h*K2, u + half_h*L2, input)
	L3 = f2(v + half_h*K2, u + half_h*L2, a)

	K4 = f1(v + h*K3, u + h*L3, input);
	L4 = f2(v + h*K3, u + h*L3, a);

	v_new = v + sixth_h * (K1 + 2*K2 + 2*K3 + K4);
	u_new = u + sixth_h * (L1 + 2*L2 + 2*L3 + L4);

	return v_new, u_new
end


# -------------------- neuron update with input after RK --------------------
f1_r(v, u) = (0.04*v + 5.0)*v + 140.0 - u
f2_r(v, u, a) = a*(0.2*v - u)

function neuronupdate_rk4_r(v, u, a, input, h) 
	half_h = 0.5*h
	sixth_h = h/6.0

	K1 = f1_r(v, u)
	L1 = f2_r(v, u, a)

	K2 = f1_r(v + half_h*K1, u + half_h*L1)
	L2 = f2_r(v + half_h*K1, u + half_h*L1, a)

	K3 = f1_r(v + half_h*K2, u + half_h*L2)
	L3 = f2_r(v + half_h*K2, u + half_h*L2, a)

	K4 = f1_r(v + h*K3, u + h*L3)
	L4 = f2_r(v + h*K3, u + h*L3, a)

	v_new = v + sixth_h * (K1 + 2*K2 + 2*K3 + K4) + input
	u_new = u + sixth_h * (L1 + 2*L2 + 2*L3 + L4)

	return v_new, u_new
end


# -------------------- input functions --------------------
expsampl(λ) = -log(rand()) / λ

function poistrain(λ, fs, N)
	train = zeros(N)
	t = expsampl(λ)
	i = Int(round(t*fs))
	while i <= N
		train[i] = 1.0
		t += expsampl(λ)
		i = Int(round(t*fs))
	end
	return train
end

################################################################################
# Main simulation
################################################################################
fs = 1000
dt = 1.0/fs
dur = 5.0
N = Int(round(dur/dt))
ts = collect(0:N-1) .* dt
vs = zeros(N)
us = zeros(N)

vs[1] = v_init
us[1] = u_init

input = poistrain(5.0, fs, N) .* 20.0

for k ∈ 2:N 
	global input
	vs[k], us[k] = neuronupdate_rk4_r(vs[k-1], us[k-1], a_inh, input[k-1], 1000.0*dt)	
	if vs[k] >= 30.0
		vs[k] = -65.0
		us[k] += d_inh
	end
end

p1 = plot(ts, vs, label="v")
plot!(p1, ts, us, label="u")
p2 = plot(ts, input, label="input")

p = plot(p1, p2, layout=(2,1))

"""
unsigned int sim_checkspiking(neuron *neurons, FLOAT_T *neuronoutputs,
								unsigned int n, FLOAT_T t, spikerecord *sr)
{
	size_t k;
	unsigned int numspikes=0;
	for (k=0; k<n; k++) {
		neuronoutputs[k] = 0.0;
		if (neurons[k].v >= 30.0) {
			sr_save_spike(sr, k, t);
			neuronoutputs[k] = 1.0;
			neurons[k].v = -65.0;
			neurons[k].u += neurons[k].d;
			numspikes += 1;
		}
	}
	return numspikes;
}


"""

"""
static inline FLOAT_T f1(FLOAT_T v, FLOAT_T u, FLOAT_T input) {
	return (0.04*v + 5.0)*v + 140.0 - u + input;
}

static inline FLOAT_T f2(FLOAT_T v, FLOAT_T u, FLOAT_T input, FLOAT_T a) {
	return a*(0.2*v - u);
}

void neuronupdate_rk4(FLOAT_T *v, FLOAT_T *u, FLOAT_T input, FLOAT_T a, FLOAT_T h) {
	FLOAT_T K1, K2, K3, K4, L1, L2, L3, L4, half_h, sixth_h;

	half_h = h*0.5;
	sixth_h = h/6.0;
	
	K1 = f1(*v, *u, input);
	L1 = f2(*v, *u, input, a);
	K2 = f1(*v + half_h*K1, *u + half_h*L1, input); 
	L2 = f2(*v + half_h*K1, *u + half_h*L1, input, a);
	K3 = f1(*v + half_h*K2, *u + half_h*L2, input); 
	L3 = f2(*v + half_h*K2, *u + half_h*L2, input, a);
	K4 = f1(*v + h*K3, *u + h*L3, input);
	L4 = f2(*v + h*K3, *u + h*L3, input, a);

	*v = *v + sixth_h * (K1 + 2*K2 + 2*K3 + K4);
	*u = *u + sixth_h * (L1 + 2*L2 + 2*L3 + L4); 
}
"""





end
