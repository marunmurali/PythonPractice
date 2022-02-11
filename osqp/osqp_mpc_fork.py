import osqp
import numpy as np
import scipy as sp
from scipy import sparse

#plot prep
import matplotlib.pyplot as plt
#plt.style.use('_mpl-gallery')
u_plot = []

# Discrete time model of a quadcopter
Ad = sparse.csc_matrix([
  [1.,      0.00254],
  [0.,      0.9661 ]
])
Bd = sparse.csc_matrix([
  [0.     ],
  [0.0315 ]])
[nx, nu] = Bd.shape

# Constraints
u0 = 0.0
umin = np.array([-6.0]) - u0
umax = np.array([10.0]) - u0
xmin = np.array([0.0, 0.0])
xmax = np.array([1.0, 9.0])

# Objective function
Q = sparse.diags([1., 0.])
QN = Q
R = 0.1*sparse.eye(1)

# Initial and reference states
x0 = np.zeros(2)
xr = np.array([1.0,0.0])

# Prediction horizon
N = 100

# Cast MPC problem to a QP: x = (x(0),x(1),...,x(N),u(0),...,u(N-1))
# - quadratic objective
P = sparse.block_diag([sparse.kron(sparse.eye(N), Q), QN,
                       sparse.kron(sparse.eye(N), R)], format='csc')
# - linear objective
q = np.hstack([np.kron(np.ones(N), -Q.dot(xr)), -QN.dot(xr),
               np.zeros(N*nu)])
# - linear dynamics
Ax = sparse.kron(sparse.eye(N+1),-sparse.eye(nx)) + sparse.kron(sparse.eye(N+1, k=-1), Ad)
Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, N)), sparse.eye(N)]), Bd)
Aeq = sparse.hstack([Ax, Bu])
leq = np.hstack([-x0, np.zeros(N*nx)])
ueq = leq
# - input and state constraints
Aineq = sparse.eye((N+1)*nx + N*nu)
lineq = np.hstack([np.kron(np.ones(N+1), xmin), np.kron(np.ones(N), umin)])
uineq = np.hstack([np.kron(np.ones(N+1), xmax), np.kron(np.ones(N), umax)])
# - OSQP constraints
A = sparse.vstack([Aeq, Aineq], format='csc')
l = np.hstack([leq, lineq])
u = np.hstack([ueq, uineq])

# Create an OSQP object
prob = osqp.OSQP()

# Setup workspace
prob.setup(P, q, A, l, u, warm_start=True)

# Simulate in closed loop
nsim = 5000
for i in range(nsim):
    # Solve
    res = prob.solve()


    # Check solver status
    if res.info.status != 'solved':
        raise ValueError('OSQP did not solve the problem!')

    # Apply first control input to the plant
    ctrl = res.x[-N*nu:-(N-1)*nu]
    x0 = Ad.dot(x0) + Bd.dot(ctrl)

    # Update initial state
    l[:nx] = -x0
    u[:nx] = -x0
    prob.update(l=l, u=u)

    #append plot data
    u_plot.append(ctrl)

x = np.linspace(0, 100, nsim)
fig, ax = plt.subplots()
ax.plot(x, u_plot, linewidth=2.0)
plt.show()