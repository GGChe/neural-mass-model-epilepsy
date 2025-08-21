from numpy import pi, sqrt
from numpy import exp
from numpy import dot
from numpy import interp


def vector_field(t,y,dy,u,tau,h,tau_v1,h_v1,tau_v2,h_v2,m_in_timed_input_in1,v_thr,s,m_max,weight_v1,v_thr_v1,s_v1,m_max_v1,m_in_input,time,source_idx_in0,weight_in0,source_idx,weight):


	v = y[0]
	x = y[1]
	v_v1 = y[2]
	x_v1 = y[3]
	v_v2 = y[4:6]
	x_v2 = y[6:8]
	m = m_max/(exp(s*(-v + v_thr - v_v1)) + 1)
	m_in_v2 = dot(weight_v1, m)
	m_in0 = m_max_v1/(exp(s_v1*(v_thr_v1 - v_v2)) + 1)
	m_in_timed_input_in1[0] = interp(t, time, m_in_input)
	m_in_in0 = weight_in0*m_in0[source_idx_in0]
	m_in_in1 = m_in_timed_input_in1
	m_in = m_in_in0 + m_in_in1
	m_in_v1 = weight*m_in0[source_idx]
	
	dy[0] = x
	dy[1] = h*(m_in + u)/tau - 2*x/tau - v/tau**2
	dy[2] = x_v1
	dy[3] = h_v1*m_in_v1/tau_v1 - 2*x_v1/tau_v1 - v_v1/tau_v1**2
	dy[4:6] = x_v2
	dy[6:8] = h_v2*m_in_v2/tau_v2 - 2*x_v2/tau_v2 - v_v2/tau_v2**2

	return dy