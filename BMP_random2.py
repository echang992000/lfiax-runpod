import omegaconf
import hydra
from hydra.core.hydra_config import HydraConfig
import wandb
import os, sys
import csv, time
import pickle as pkl
import math
import random

import torch
from torch.distributions import Uniform, TransformedDistribution, Distribution
from torch.distributions.transforms import Transform
from torch.distributions import constraints

import numpy as np
from functools import partial

import haiku as hk

from lfiax.utils.minebed_oed import sdm_minebed

from bmp_simulator.simulate_bmp import bmp_simulator

from sbi.diagnostics.lc2st import LC2ST

from typing import (
    Any,
    Iterator,
    Mapping,
    Optional,
    Tuple,
    Callable,
)


class LogUniform(Transform):
    """
    Defines a transformation for a log-uniform distribution.
    """
    bijective = True
    sign = +1  # Change to -1 if the transform is decreasing in the interval

    def __init__(self, low, high):
        super().__init__()
        self.low = low
        self.high = high

    def _call(self, x):
        return torch.exp(x * (self.high - self.low) + self.low)

    def _inverse(self, y):
        return (torch.log(y) - self.low) / (self.high - self.low)

    def log_abs_det_jacobian(self, x, y):
        return (self.high - self.low) * x + self.low

    @property
    def domain(self):
        return constraints.interval(0.0, 1.0)

    @property
    def codomain(self):
        return constraints.positive


def make_torch_bmp_prior():
    low = torch.log(torch.tensor(1e-6))
    high = torch.log(torch.tensor(1.0))

    uniform = Uniform(low=torch.tensor(1e-6), high=torch.tensor(1.0))

    log_uniform = TransformedDistribution(uniform, LogUniform(low, high))

    return log_uniform


class MultiLogUniform(Distribution):
    """
    A class that represents multiple independent log-uniform distributions.
    """
    def __init__(self, num_priors):
        super().__init__()
        self.priors = [make_torch_bmp_prior() for _ in range(num_priors)]

    def sample(self, sample_shape=torch.Size()):
        return torch.stack([prior.sample(sample_shape) for prior in self.priors], dim=-1)


class Workspace:
    def __init__(self, cfg):
        self.cfg = cfg

        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')

        current_time = time.localtime()
        current_time_str = f"{current_time.tm_year}.{current_time.tm_mon:02d}.{current_time.tm_mday:02d}.{current_time.tm_hour:02d}.{current_time.tm_min:02d}"
        
        self.subdir = os.path.join(os.getcwd(), 'icml_2024', 'BMP_minebed', str(cfg.designs.num_xi), str(cfg.seed), current_time_str)
        os.makedirs(self.subdir, exist_ok=True)

        # Torch seed stuff
        seed = cfg.seed
        self.seed = seed
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # torch.cuda.manual_seed(seed)
        # torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        # torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.deterministic = True
        

    def run(self) -> Callable:
        tic = time.time()
        logf, writer = self._init_logging()

        prng_seq = hk.PRNGSequence(self.seed)
        
        num_priors = 2
        priors = MultiLogUniform(num_priors)

        # Simulator (BMP onestep model) to use
        simulator = partial(
            bmp_simulator, 
            model_size=(1,1,1),
            model='onestep', 
            fixed_receptor=True)

        num_design_rounds = 3
        y_obs = None
        DATASIZE = 5_000
        BATCHSIZE = DATASIZE
        BO_MAX_NUM = 1
        NN_layers = 1
        NN_hidden = 150
        design_dims = self.cfg.designs.num_xi
        
        thetas = priors.sample((DATASIZE,)).numpy()
        prior_thetas = thetas

        # Just modify this for random. Don't need speed, so just rewrite with random design lol
        xi = np.random.uniform(low=0., high=1e3, size=(1, 1))
        total_EIG = 0
        xis = []
        xis.append(xi)

        for design_round in range(num_design_rounds):

            bed_obj = sdm_minebed(
                simulator = simulator,
                params = priors,
                y_obs = y_obs,
                DATASIZE = DATASIZE,
                BATCHSIZE = BATCHSIZE,
                BO_MAX_NUM = BO_MAX_NUM, 
                dims = design_dims,
                NN_layers = NN_layers,
                NN_hidden = NN_hidden,
                prior_sims = thetas,
                )
            
            bed_obj.train_final_model(n_epoch=20_000, batch_size=DATASIZE)

            bed_obj.model.eval()
            # TODO: Change this
            bmp_output = simulator(xi, thetas)

            EIGs = bed_obj.model(torch.from_numpy(thetas), torch.from_numpy(bmp_output))

            EIG = torch.mean(EIGs).detach().numpy()
            
            inference_time = time.time() - tic

            # Compute weights
            T = bed_obj.model(torch.from_numpy(thetas), torch.from_numpy(bmp_output)).data.numpy().reshape(-1)
            post_weights_1 = np.exp(T - 1)
            ws_norm_1 = post_weights_1 / np.sum(post_weights_1)

            K = 100_000
            idx_samples = random.choices(range(len(ws_norm_1)), weights=ws_norm_1, k=K)
            post_samples_1 = thetas[idx_samples]

            thetas = post_samples_1[:DATASIZE]

            writer.writerow({
                    'seed': self.seed,
                    'EIG': EIG,
                    'opt_design': bed_obj.d_opt,
                    'inference_time':float(inference_time)
                })
            logf.flush()

            # Reset random design
            xi = np.random.uniform(low=0., high=1e3, size=(1, 1))
            xis.append(xi)
            total_EIG += EIG
        
        ############# Record LC2ST Metrics #############
        x_sbi = simulator(np.array(xis), thetas)
        if self.device == "cuda":
            x_o = torch.from_numpy(np.asarray(self.static_outputs[0,:][None,:])).cuda()
            post_samples_torch = torch.from_numpy(thetas).cuda()
            xs = torch.from_numpy(np.array(x_sbi)).cuda()
            thetas = torch.from_numpy(prior_thetas).cuda()
        else:
            x_o = torch.from_numpy(np.asarray(self.static_outputs[0,:][None,:])).float()
            post_samples_torch = torch.from_numpy(thetas).float()
            xs = torch.from_numpy(np.array(x_sbi)).float()
            thetas = torch.from_numpy(prior_thetas).float()
        
        lc2st = LC2ST(
            thetas=thetas,
            xs=xs,
            posterior_samples=post_samples_torch[:xs.shape[0]],
            seed=self.seed,
            num_folds=1,
            num_ensemble=1,
            classifier="mlp",
            z_score=False,
            num_trials_null=100,
            permutation=True,
        )

        lc2st.train_under_null_hypothesis()
        lc2st.train_on_observed_data()
        
        theta_o = post_samples_torch
        breakpoint()
        statistic = lc2st.get_statistic_on_observed_data(theta_o=theta_o, x_o=x_o)
        print("L-C2ST statistic on observed data:", statistic)
        p_value = lc2st.p_value(theta_o=theta_o, x_o=x_o)
        print("P-value for L-C2ST:", p_value)

        # Decide whether to reject the null hypothesis at a significance level alpha
        alpha = 0.05  # 95% confidence level
        reject = lc2st.reject_test(theta_o=theta_o, x_o=x_o, alpha=alpha)
        print(f"Reject null hypothesis at alpha = {alpha}:", reject)
        del lc2st
        
        # objects = {
        #     'post_samples': post_samples_1,
        # }
        # with open(f"{self.subdir}/posts_bed_obj.pkl", "wb") as f:
        #     pkl.dump(objects, f)
        #     bed_obj.save(f"{self.subdir}/bed_obj")
        

    def _init_logging(self):
        path = os.path.join(self.subdir, 'log.csv')
        logf = open(path, 'a') 
        fieldnames = ['seed', 'EIG', 'opt_design', 'inference_time']
        writer = csv.DictWriter(logf, fieldnames=fieldnames)
        if os.stat(path).st_size == 0:
            writer.writeheader()
            logf.flush()
        return logf, writer
        

from BMP_random2 import Workspace as W

@hydra.main(version_base=None, config_path=".", config_name="config_bmp")
def main(cfg):
    fname = os.getcwd() + '/latest.pt'
    if os.path.exists(fname):
        print(f'Resuming fom {fname}')
        with open(fname, 'rb') as f:
            workspace = pkl.load(f)
        print(f"STEP: {workspace.step:5d}; Xi: {workspace.xi};\
             Xi Grads: {workspace.xi_grads}; Loss: {workspace.loss}")
    else:
        workspace = W(cfg)

    workspace.run()


if __name__ == "__main__":
    main()
