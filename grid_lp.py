#!/usr/bin/env python
import pickle
import numpy as np
import itertools
from scipy.io import netcdf

from cvxopt import matrix, solvers
solvers.options['show_progress'] = True

import PyGMO

from utils import Timer

def powerLoss(kms):
    #assuming 800kV dC powerlines - approximataly 3.0% loss
    # 2.0% is loss of convertors and begin/end
    lossfrac = 2.0 + (6.0 / 2000.) * kms
    return lossfrac / 100.


def read_data():
    DAY = 0
    with open("alldata.pk", "rb") as f:
        data = pickle.load(f)

    data["irradiance"] = data["irradiance"][DAY]
    data["clouds"] = data["clouds"][DAY]
    
    # reformat cloud data
    rep = data["irradiance"].shape[-1]/data["clouds"].shape[-1]
    clouds = np.zeros(data["irradiance"].shape)
    for t in range(data["irradiance"].shape[-1]):
        clouds[:, t] = data["clouds"][:, t//rep]
    data["clouds"] = clouds
    
    # powerloss
    data["powerloss"] = np.array(map(powerLoss, data["distance"]))
    return data

    
class grid_linear_problem(PyGMO.problem.base):

    __data = read_data()

    def __init__(self):
        self.data = self.__data

        # flow link indecies
        ids = range(len(self.data["zones"]))
        self.fidx = list(itertools.product(ids, ids))

        for item in [(x, x) for x in ids]:
            del self.fidx[self.fidx.index(item)]

        # dimensions
        self.N = len(self.data["zones"])
        self.n = len(self.fidx)
        dim = self.n / 2

        super(grid_linear_problem, self).__init__(dim, dim, 2, 0, 0, 0)
        self.set_bounds(0, 1)
        
    def power_prod(self, cloud, irradiance):
        # area in 1e6 km^2
        return .2 * irradiance * cloud / 1e6

    def get_total_dist(self, flow):
        dist = 0
        upper_idx = np.triu_indices(self.N, k=1)
        upper_idx = zip(upper_idx[0], upper_idx[1])
        for i in np.nonzero(flow)[0]:
            dist += self.data["distance"][upper_idx[i][0], upper_idx[i][1]]
        return dist

    def get_flow_matrix(self, x, t):
        flow = x[self.N+t*self.n:(t+1)*self.n]
        res = np.zeros()
        for f, (i, j) in enumerate(self.fidx):
            res[i, j] = f
        return res
        
    def create_sub_matrix(self, t, flow):
        N = len(self.data["zones"])
        
        # calculate production
        cloud = self.data["clouds"][:, t]
        irradiance = self.data["irradiance"][:, t]
        # TODO: should be moved to init
        production_coeff = np.array(map(lambda x: self.power_prod(*x), zip(cloud, irradiance)))

        # demand
        demand = self.data["demand"][:, t]  # in peta W

        n = len(self.fidx)  # flow indices
    
        # A
        A_balance = np.zeros((N, N+n))
        for i, _ in enumerate(self.data["zones"]):
            # production
            A_balance[i, i] = -1 * production_coeff[i] 
            
            # flow
            for j, (f_from, f_to) in enumerate(self.fidx):
                if flow[j]:
                    if i == f_from:
                        # outgoing
                        A_balance[i, N+j] = 1
                    if i == f_to:
                        # incoming
                        A_balance[i, N+j] = -self.data["powerloss"][f_from, f_to]
                    
        A_area_pos = np.hstack([np.identity(N), np.zeros((N, n))]) * -1
        A_link_pos = np.hstack([np.zeros((n, N)), np.identity(n)]) * -1
        A = np.vstack([A_area_pos, A_link_pos, A_balance])
        
        # b
        b_area_pos = np.array([0] * N)
        b_link_pos = np.array([0] * n)
        b_balance = np.array(demand) * -1  # -p <= -d  <=>  p >= d
        b = np.hstack([b_area_pos, b_link_pos, b_balance])
        
        return (A, b)
        
    def solve(self, flow):
        N = len(self.data["zones"])
        As_area = []
        As_flow = []
        bs = []
        for t, _ in enumerate(self.data["time"]):
            (A, b) = self.create_sub_matrix(t, flow)
            # seperate area and flow part of sub problem
            A = A[N:, :]  # remove area_pos part
            A_area = A[:, 0:N]  # split in area
            A_flow = A[:, N:]   # ... and flow
            As_area.append(A_area)
            As_flow.append(A_flow)

            b = b[N:] # remove area_pos part
            bs.append(b)

        from scipy.linalg import block_diag
        A_flow = block_diag(*As_flow)
        A_flow = np.vstack([np.zeros((N, A_flow.shape[1])), A_flow])  # add N rows of 0's on top

        A_area = np.vstack(As_area)
        A_area = np.vstack([np.identity(N) * -1, A_area])  # add area_pos constraint

        A = np.hstack([A_area, A_flow])
        
        b = np.concatenate(bs)
        b = np.concatenate([np.array([0] * N), b])  # add area_pos constraints

        c = np.array([1.] * N + [0.] * (A.shape[1]-N))

        print "min: c[%s] * x subj. to A[%s] * x <= b[%s]" % (str(c.shape), str(A.shape), str(b.shape))
                
        # solve
        A = matrix(A)
        b = matrix(b)
        c = matrix(c)
        sol = solvers.lp(c, A ,b)
        return sol

    def _objfun_impl(self, x):
        # transform upper triangular flow indices to full list
        upper_idx = np.triu_indices(self.N, k=1)
        upper_idx = zip(upper_idx[0], upper_idx[1])
    
        flow = [False] * self.n
        for (f, (i, j)) in zip(x, upper_idx):
            if f:
                flow[self.fidx.index((i, j))] = True
                flow[self.fidx.index((j, i))] = True
                
        sol = self.solve(flow)

        if sol["status"] == "optimal":
            # objectives
            f_area = sum(sol["x"][:self.N]) 
            f_links = self.get_total_dist(x)
        else:
            f_area = float("inf")
            f_links = float("inf")

        print "    area: %.2f    length: %.2f    links: %d" % (f_area, f_links, len(np.nonzero(x)[0]))
        return (f_area, f_links)
        
if __name__ == "__main__":
    prob = grid_linear_problem()
    print prob
    
#    # fully connected
#    pop = PyGMO.population(prob, 0)
#    pop.push_back([True] * prob.dimension)
#    print pop.champion.f

    # random population, one fully connected
    pop = PyGMO.population(prob, 19)
    pop.push_back([True] * prob.dimension)
    algo = PyGMO.algorithm.ihs(1)
    for i in xrange(100):
        with Timer(verbose=True) as t:
            pop = algo.evolve(pop)

        # save generation
        fname = "results/gen-%d.pickle" % i
        with open(fname, "wb") as f:
            pickle.dump(pop, f)
