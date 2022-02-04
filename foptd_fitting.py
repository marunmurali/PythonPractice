%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import control
from scipy.optimize import minimize

def foptd(t, K=1, tau=1, tau_d=0):
    tau_d = max(0,tau_d)
    tau = max(0,tau)
    return np.array([K*(1-np.exp(-(t-tau_d)/tau)) if t >= tau_d else 0 for t in t])

t = np.linspace(0,50,200)
tau = 10
tau_delay = 3
K = 2

y = foptd(t,K,tau,tau_delay)
plt.plot(t,y)
plt.xlabel('Time [min]')
plt.ylabel('Response')
plt.title('FOPTD Step Response')

# create the hypothetical problem data
delta_y,t = control.step(0.05*control.tf([-2, 1],[25, 10, 1]))
y = 0.87 + delta_y
t = t + 60

plt.plot(t,y)
plt.xlabel('Time [min]')
plt.ylabel('mole fraction')
plt.title('Vapor mole fraction response to 10 kg/hr increase in condensor coolng')

ts = t - t[0]
ys = (y - y[0])/(120 - 110)

plt.plot(ts, ys)
plt.title('Shifted and Scaled Data')

z = foptd(ts, 0.005, 10, 3)
plt.plot(ts,ys,ts,z)
plt.legend(['Experiment','FOPTD'])

def err(X,t,y):
    K,tau,tau_d = X
    z = foptd(t,K,tau,tau_d)
    iae = sum(abs(z-y))*(max(t)-min(t))/len(t)
    return iae

X = [0.005,10,3]
err(X,ts,ys)

K,tau,tau_d = minimize(err,X,args=(ts,ys)).x

print("K = {:.5f}".format(K))
print("tau = {:.2f}".format(tau))
print("tau_d = {:.2f}".format(tau_d))

z = foptd(ts,K,tau,tau_d)
ypred = y[0] + z*(120 - 110)

plt.plot(t,y,t,ypred)
plt.xlabel('Time [min]')
plt.ylabel('vapor mole fraction')
plt.legend(['Experiment','FOPTD'])