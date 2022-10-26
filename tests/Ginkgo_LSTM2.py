import pyprob
import numpy as np
import ot
import torch
import pickle
import cProfile

from pyprob.dis import ModelDIS
from showerSim import invMass_ginkgo
from torch.utils.data import DataLoader
from pyprob.nn.dataset import OnlineDataset
from pyprob.util import InferenceEngine
from pyprob.util import to_tensor
from pyprob import Model
from pyprob.model import Parallel_Generator
import math
from pyprob.distributions import Normal
from pyprob.distributions.delta import Delta


import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as mpl_cm
plt.ion()

import sklearn as skl
from sklearn.linear_model import LinearRegression

from geomloss import SamplesLoss
sinkhorn = SamplesLoss(loss="sinkhorn", p=1, blur=.05)
def sinkhorn_t(x,y):
    x = to_tensor(x)
    y = torch.stack(y)
    return sinkhorn(x,y)

def ot_dist(x,y):
    # x = to_tensor(x)
    # y = torch.stack(y)
    x = np.array(x)
    y = np.array(torch.stack(y))
    a = ot.unif(len(x))
    b = ot.unif(len(y))
    Mat = ot.dist(x, y, metric='euclidean')
    #Mat1 /= Mat1.max()
    distance = to_tensor(ot.emd2(a,b,Mat))
    return distance


device = "cpu"

from pyprob.util import set_device
set_device(device)

#rate = [5,9.5]
obs_leaves = to_tensor([[ 25.8005,  15.8486,  13.7905,  14.9743],
        [ 64.6767,  39.8886,  34.8096,  37.1519],
        [112.1183,  68.9756,  59.4660,  65.3958],
        [ 17.2854,   9.8922,   9.2324,  10.7553],
        [  8.0760,   4.8046,   4.1940,   4.9540],
        [ 32.6218,  17.6533,  21.0402,  17.6020],
        [ 78.1670,  42.4337,  50.0361,  42.4953],
        [ 23.3093,  12.4575,  14.6925,  13.1246],
        [  8.5133,   4.5174,   5.0757,   5.1276],
        [ 17.5119,   8.7775,  10.6228,  10.8064],
        [ 11.6579,   5.0303,   7.2342,   7.6330],
        [  0.8138,   0.3167,   0.5029,   0.5491],
        [  0.5715,   0.3441,   0.2433,   0.3707]])


QCD_mass = to_tensor(30.)
#rate=to_tensor([QCD_rate,QCD_rate]) #Entries: [root node, every other node] decaying rates. Choose same values for a QCD jet
jetdir = to_tensor([1.,1.,1.])
jetP = to_tensor(400.)
jetvec = jetP * jetdir / torch.linalg.norm(jetdir) ## Jetvec is 3-momentum. JetP is relativistic p.


# Actual parameters
pt_min = to_tensor(0.3**2)
M2start = to_tensor(QCD_mass**2)
jetM = torch.sqrt(M2start) ## Mass of initial jet
jet4vec = torch.cat((torch.sqrt(jetP**2 + jetM**2).reshape(-1), jetvec))
minLeaves = 1
maxLeaves = 10000 # unachievable, to prevent rejections
maxNTry = 100



class SimulatorModelDIS(invMass_ginkgo.SimulatorModel, ModelDIS):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def dummy_bernoulli(self, jet):
        return True

    def forward(self, inputs=None):
        assert inputs is None # Modify code if this ever not met?
        # Sample parameter of interest from Unif(0,10) prior
        root_rate = pyprob.sample(pyprob.distributions.Uniform(0.01, 10.),
                                  name="decay_rate_parameter")
        decay_rate = pyprob.sample(pyprob.distributions.Uniform(0.01, 10.),
                                   name="decay_rate_parameter")
        # Simulator code needs two decay rates for (1) root note (2) all others
        # For now both are set to the same value
        inputs = [root_rate, decay_rate]
        jet = super().forward(inputs)
        delta_val = self.dummy_bernoulli(jet)
        bool_func_dist = pyprob.distributions.Bernoulli(delta_val)
        pyprob.observe(bool_func_dist, name = "dummy")
        return jet

# Make instance of the simulator
simulatorginkgo = SimulatorModelDIS(jet_p=jet4vec,  # parent particle 4-vector
                                    pt_cut=float(pt_min),  # minimum pT for resulting jet
                                    Delta_0= M2start,  # parent particle mass squared -> needs tensor
                                    M_hard=jetM,  # parent particle mass
                                    minLeaves=1,  # minimum number of jet constituents
                                    maxLeaves=10000,  # maximum number of jet constituents (a large value to stop expensive simulator runs)
                                    suppress_output=True,
                                    obs_leaves=obs_leaves,
                                    dist_fun=sinkhorn_t)


simulatorginkgo.load_inference_network('New_LSTM_Network_20')

for i in range(1,5):
    simulatorginkgo.train(iterations=20, importance_sample_size=5000)
    simulatorginkgo.save_inference_network(f'New_LSTM_Network_{(i+1)*20}')
    posterior = simulatorginkgo.posterior(num_traces=2000, inference_engine=InferenceEngine.DISTILLING_IMPORTANCE_SAMPLING, observe={'dummy':1})
    posterior = simulatorginkgo.update_DIS_weights(posterior)
    with open(f'LSTM_595_posterior_{(i+1)*20}', 'wb') as f:
        pickle.dump(posterior, f)
