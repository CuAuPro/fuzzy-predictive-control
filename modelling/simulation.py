from control.matlab import *
import numpy as np
from .utils import ShiftingArray

def simulate_first_order_system(sys, amplitudes, times):
    input_signal = np.column_stack((times, amplitudes))

    y, t, _ = lsim(sys, input_signal[:, 1], input_signal[:, 0])
    
    measurements = {'time': t, 'response': y}
    return measurements

def simulate_discrete_state_space(A, B, C, D, x0, u):
    num_samples = len(u)
    num_states = A.shape[0]
    num_outputs = C.shape[0]

    # Initialize arrays to store the system response
    x = np.zeros((num_samples + 1, num_states))
    y = np.zeros((num_samples, num_outputs))

    # Set the initial state
    x[0, :] = x0

    # Simulate the system
    for k in range(num_samples):
        x[k + 1, :] = A.dot(x[k, :]) + B.dot(u[k, :])
        y[k, :] = C.dot(x[k, :]) + D.dot(u[k, :])

    return y


class Helicrane():
    
    def __init__(self, m=0.45, g=9.81, l=0.56, d_static=0.6):
        self.m = m # Mass of the helicopter
        self.g = g # Mass of the helicopter
        self.l = l # Length of the crane arm
        self.d_static = d_static # Damping coefficient
        self.x = None
        
    def dynamics(self, states, T):
        theta, omega = states
        theta_dot = omega
        
        # Nonlinear damping coefficient
        d = self.d_static + 1.5 * np.sin(theta)
        omega_dot = (T - d * omega - self.m * self.g * self.l * np.sin(theta) - 5*theta**3) / (self.m * self.l**2)
        return [theta_dot, omega_dot]
    
    
    def simulate(self, x0, u, t):

        num_samples = len(u)
        num_states = len(x0)

        # Initialize arrays to store the system response
        x = np.zeros((num_samples, num_states))
        x[0, :] = x0

        # Simulate the system
        for i in range(1, num_samples):
            dt = t[i] - t[i - 1]
            x[i, :] = x[i - 1, :] + np.array(self.dynamics(x[i - 1, :], u[i - 1])) * dt
        return x
    
    def reset_states(self, x0):
        self.x = x0
        
    def step(self, u, dt=0.01):
        self.x = self.x + np.array(self.dynamics(self.x, u)) * dt
        return self.x


def fuzzy_sim(G, gk, U, mi=1):
    nr_poles = np.ravel(G[0].den).shape[0]-1
    nr_zeros = np.ravel(G[0].num).shape[0]
    y_model = np.zeros(U.shape[0]+max(nr_poles, nr_zeros)+1)
    num = np.zeros((gk.c, nr_zeros))
    den = np.zeros((gk.c, nr_poles))

    idx_start_num = nr_poles - nr_zeros + 1
    for i in range(gk.c):
        num[i,:] = np.ravel(G[i].num)[::-1]
        den[i,:] = np.ravel(G[i].den)[1:][::-1]

    start_idx = max(nr_zeros, nr_poles)
    for sample in range(start_idx, U.shape[0]):
        beta = np.ones((gk.c))
        y_l = np.zeros((gk.c, 1))
        for i in range(gk.c):
            variances = np.sort(np.diag(gk.A[i]))[::-1]
            vari = variances[0]
            v_l = gk.V[i,0]
            x_l = U[sample]
            beta[i] = beta[i]*np.exp(-0.5*(v_l-x_l)**2/(mi*vari))
        beta = beta/np.sum(beta)
        
        for i in range(gk.c):
            a = np.concatenate([den[i,:],
                        num[i,:],
                        [np.sum(den[i,:])+1],
                        [np.sum(num[i,:])]]).reshape(1,-1)

            b = np.concatenate([-y_model[sample-nr_poles:sample],
                            U[sample-nr_zeros:sample],
                            [gk.V[i,1]],
                            [-gk.V[i,0]]]).reshape(-1,1)

            y_l[i] = np.dot(a,b).ravel()
                                        
        y_model[sample] = np.dot(beta, y_l)
    y_model = y_model[:-start_idx-1]

    return y_model

class FuzzyModel:

    def __init__(self, G, gk, mi=1):
        self.G = G
        self.gk = gk
        self.mi=mi

        nr_poles = np.ravel(self.G[0].den).shape[0]-1
        nr_zeros = np.ravel(self.G[0].num).shape[0]

        # Create an instance of the ShiftingArray class with size 3
        self.y_lag = ShiftingArray(nr_poles)
        self.u_lag = ShiftingArray(nr_zeros)

        self.num = np.zeros((gk.c, nr_zeros))
        self.den = np.zeros((gk.c, nr_poles))
        for i in range(gk.c):
            self.num[i,:] = np.ravel(self.G[i].num)[::-1]
            self.den[i,:] = np.ravel(self.G[i].den)[1:][::-1]

        
    def step(self, u):   
        self.u_lag.add_element(u)
     
        beta = np.ones((self.gk.c))
        y_l = np.zeros((self.gk.c, 1))
        for i in range(self.gk.c):
            variances = np.sort(np.diag(self.gk.A[i]))[::-1]
            vari = variances[0]
            v_l = self.gk.V[i,0]
            x_l = u
            beta[i] = beta[i]*np.exp(-0.5*(v_l-x_l)**2/(self.mi*vari))
        beta = beta/np.sum(beta)
        
        for i in range(self.gk.c):
            a = np.concatenate([self.den[i,:],
                        self.num[i,:],
                        [np.sum(self.den[i,:])+1],
                        [np.sum(self.num[i,:])]]).reshape(1,-1)

            b = np.concatenate([-self.y_lag.array,
                            self.u_lag.array,
                            [self.gk.V[i,1]],
                            [-self.gk.V[i,0]]]).reshape(-1,1)

            y_l[i] = np.dot(a,b).ravel()
                                        
        y = np.dot(beta, y_l)

        self.y_lag.add_element(y)

        
        return y
    

class FuzzyModelSS:

    def __init__(self, G_ss, gk, mi=1):
        self.G_ss = G_ss
        self.gk = gk
        self.mi=mi

        nr_poles = len(np.diag(self.G_ss[0]['A']))
        nr_zeros = len(np.diag(self.G_ss[0]['B']))

        self.x = np.zeros_like(np.diag(self.G_ss[0]["A"])).reshape(-1,1)
        self.x_curr = np.zeros_like(np.diag(self.G_ss[0]["A"])).reshape(-1,1)
        # Create an instance of the ShiftingArray class with size 3
        self.y_lag = ShiftingArray(nr_poles)
        self.u_lag = ShiftingArray(nr_zeros)


        self.Am = np.zeros_like(self.G_ss[0]["A"])
        self.Bm = np.zeros_like(self.G_ss[0]["B"])
        self.Cm = np.zeros_like(self.G_ss[0]["C"])
        self.Dm = np.zeros_like(self.G_ss[0]["D"])
        self.Rm = np.zeros_like(self.G_ss[0]["R"])

        
    def step(self, u):   
        self.u_lag.add_element(u)

        beta = np.ones((self.gk.c))
        for i in range(self.gk.c):
            variances = np.sort(np.diag(self.gk.A[i]))[::-1]
            vari = variances[0]
            v_l = self.gk.V[i,0]
            x_l = u
            beta[i] = beta[i]*np.exp(-0.5*(v_l-x_l)**2/(self.mi*vari))
        beta = beta/np.sum(beta)

        self.Am = np.zeros_like(self.G_ss[0]["A"])
        self.Bm = np.zeros_like(self.G_ss[0]["B"])
        self.Cm = np.zeros_like(self.G_ss[0]["C"])
        self.Dm = np.zeros_like(self.G_ss[0]["D"])
        self.Rm = np.zeros_like(self.G_ss[0]["R"])

        for i in range(self.gk.c):
            self.Am = self.Am + self.G_ss[i]["A"]*beta[i]
            self.Bm = self.Bm + self.G_ss[i]["B"]*beta[i]
            self.Cm = self.Cm + self.G_ss[i]["C"]*beta[i]
            self.Dm = self.Dm + self.G_ss[i]["D"]*beta[i]
            self.Rm = self.Rm + self.G_ss[i]["R"]*beta[i]
    
        y = self.Cm@self.x
        self.x_curr = self.x
        self.x = self.Am@self.x + self.Bm*u + self.Rm.reshape(-1,1)

        self.y_lag.add_element(y)

        
        return y



class ModelSS:

    def __init__(self, G_ss):
        self.G_ss = G_ss

        nr_poles = len(np.diag(self.G_ss.A))
        nr_zeros = len(np.diag(self.G_ss.B))

        self.x = np.zeros_like(np.diag(self.G_ss.A))
        # Create an instance of the ShiftingArray class with size 3
        self.y_lag = ShiftingArray(nr_poles)
        self.u_lag = ShiftingArray(nr_zeros)

        
    def step(self, u):   
        self.u_lag.add_element(u)

        y = self.G_ss.C@self.x
        self.x = self.G_ss.A@self.x + self.G_ss.B@np.array([u])

        self.y_lag.add_element(y)
        return y