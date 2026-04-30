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

        if cfg.wandb.use_wandb:
            wandb_config = omegaconf.OmegaConf.to_container(
                cfg, resolve=True, throw_on_missing=True
            )
            wandb.init(
                entity=self.cfg.wandb.entity,
                project=self.cfg.wandb.project,
                config=wandb_config,
            )

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

        num_priors = 2
        priors = MultiLogUniform(num_priors)

        # Simulator (BMP onestep model) to use
        simulator = partial(
            bmp_simulator,
            model_size=(1,1,1),
            model='onestep',
            fixed_receptor=True)

        minebed_cfg = self.cfg.minebed
        num_design_rounds = int(self.cfg.experiment.design_rounds)
        y_obs = None
        BO_INIT_NUM = int(minebed_cfg.bo_init_num)
        BO_MAX_NUM = int(minebed_cfg.bo_max_num)
        configured_total_sim_budget = int(minebed_cfg.total_sim_budget)
        sim_call_budget = minebed_cfg.get("sim_call_budget", None)
        if sim_call_budget is not None:
            sim_call_budget = int(sim_call_budget)
            if sim_call_budget < BO_INIT_NUM:
                raise ValueError("minebed.sim_call_budget must be >= minebed.bo_init_num")
            BO_MAX_NUM = sim_call_budget - BO_INIT_NUM
        else:
            sim_call_budget = BO_INIT_NUM + BO_MAX_NUM
        datasize = minebed_cfg.get("datasize", None)
        if datasize is None:
            DATASIZE = configured_total_sim_budget // sim_call_budget
        else:
            DATASIZE = int(datasize)
        bo_objective_evals = BO_INIT_NUM + BO_MAX_NUM
        bo_training_sim_samples = bo_objective_evals * DATASIZE
        estimated_total_sim_samples = (
            num_design_rounds * (bo_objective_evals + 2) * DATASIZE
            + DATASIZE
            + 1
        )
        print(
            "MINEBED budget: "
            f"bo_objective_evals={bo_objective_evals}, "
            f"DATASIZE={DATASIZE}, "
            f"bo_training_sim_samples={bo_training_sim_samples}, "
            f"configured_total_sim_budget={configured_total_sim_budget}"
        )
        batchsize = minebed_cfg.get("batchsize", None)
        BATCHSIZE = DATASIZE if batchsize is None else int(batchsize)
        gpyopt_eps = -np.inf if minebed_cfg.disable_gpyopt_early_stop else float(minebed_cfg.gpyopt_eps)
        N_EPOCH = int(minebed_cfg.n_epoch)
        FINAL_N_EPOCH = int(minebed_cfg.final_n_epoch)
        NN_layers = int(minebed_cfg.nn_layers)
        NN_hidden = int(minebed_cfg.nn_hidden)
        design_dims = self.cfg.designs.num_xi

        thetas = priors.sample((DATASIZE,)).numpy()
        prior_thetas = thetas
        xis = []
        x_obs = []
        EIGs= []
        round_wall_times = []

        for design_round in range(num_design_rounds):
            round_tic = time.time()

            bed_obj = sdm_minebed(
                simulator = simulator,
                params = priors,
                y_obs = y_obs,
                DATASIZE = DATASIZE,
                BATCHSIZE = BATCHSIZE,
                N_EPOCH = N_EPOCH,
                BO_INIT_NUM = BO_INIT_NUM,
                BO_MAX_NUM = BO_MAX_NUM,
                GPYOPT_EPS = gpyopt_eps,
                dims = design_dims,
                NN_layers = NN_layers,
                NN_hidden = NN_hidden,
                prior_sims = thetas,
                )

            bed_obj.train_final_model(n_epoch=FINAL_N_EPOCH, batch_size=DATASIZE)

            bed_obj.model.eval()

            bmp_output = simulator(bed_obj.d_opt[None,:], thetas)

            bed_EIGs = bed_obj.model(torch.from_numpy(thetas), torch.from_numpy(bmp_output))

            EIG = torch.mean(bed_EIGs).detach().numpy()

            inference_time = time.time() - tic

            # Compute weights
            T = bed_obj.model(torch.from_numpy(thetas), torch.from_numpy(bmp_output)).data.numpy().reshape(-1)
            post_weights_1 = np.exp(T - 1)
            ws_norm_1 = post_weights_1 / np.sum(post_weights_1)

            K = 100_000
            idx_samples = random.choices(range(len(ws_norm_1)), weights=ws_norm_1, k=K)
            post_samples_1 = thetas[idx_samples]

            thetas = post_samples_1[:DATASIZE]
            xis.append(bed_obj.d_opt)
            x_obs.append(bmp_output)
            EIGs.append(EIG)

            writer.writerow({
                    'seed': self.seed,
                    'EIG': float(EIG),
                    'opt_design': bed_obj.d_opt,
                    'inference_time':float(inference_time)
                })
            logf.flush()
            round_wall_times.append(time.time() - round_tic)

        total_EIG = float(np.array(EIGs).sum())
        print("EIG")
        print(total_EIG)
        ############# Record LC2ST Metrics #############
        
                # if self.device == "cuda":
        #     x_o = torch.from_numpy(np.asarray(self.static_outputs[0,:][None,:])).cuda()
        #     post_samples_torch = torch.from_numpy(thetas).cuda()
        #     xs = torch.from_numpy(np.array(x_sbi)).cuda()
        #     thetas = torch.from_numpy(prior_thetas).cuda()
        # else:

        x_o = simulator(np.array(xis).T, np.array([[0.85, 0.85]]))
        x_sbi = simulator(np.array(xis).T, thetas)
        # x_o = torch.from_numpy(np.asarray(np.array(x_obs).T)).float()
        post_samples_torch = torch.from_numpy(thetas).float()
        xs = torch.from_numpy(np.array(x_sbi)).float()
        thetas = torch.from_numpy(prior_thetas).float()

        lc2st = LC2ST(
            thetas=thetas,
            xs=xs,
            posterior_samples=post_samples_torch[:xs.shape[0]],
            seed=42,
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
        statistic = lc2st.get_statistic_on_observed_data(theta_o=theta_o, x_o=torch.tensor(x_o))
        median_distance = torch.median(torch.linalg.vector_norm(torch.tensor(x_sbi) - torch.tensor(x_o), axis=1))
        print("median distance: ", median_distance)
        print("L-C2ST statistic on observed data:", statistic)
        p_value = lc2st.p_value(theta_o=theta_o, x_o=torch.tensor(x_o))
        print("P-value for L-C2ST:", p_value)

        # Decide whether to reject the null hypothesis at a significance level alpha
        alpha = 0.05  # 95% confidence level
        reject = lc2st.reject_test(theta_o=theta_o, x_o=torch.tensor(x_o), alpha=alpha)
        print(f"Reject null hypothesis at alpha = {alpha}:", reject)
        if self.cfg.wandb.use_wandb:
            statistic_value = float(np.asarray(statistic).item())
            p_value_value = float(np.asarray(p_value).item())
            median_distance_value = float(median_distance.detach().cpu().item())
            wandb.log({
                "minebed/final_total_EIG": total_EIG,
                "minebed/final_inference_time": float(inference_time),
                "minebed/round_wall_time_total": float(np.sum(round_wall_times)),
                "minebed/round_wall_time_mean": float(np.mean(round_wall_times)),
                "minebed/round_wall_time_max": float(np.max(round_wall_times)),
                "minebed/lc2st_statistic": statistic_value,
                "minebed/lc2st_p_value": p_value_value,
                "minebed/lc2st_reject": bool(reject),
                "minebed/lc2st_alpha": alpha,
                "minebed/median_distance": median_distance_value,
                "minebed/bo_init_num": BO_INIT_NUM,
                "minebed/bo_max_num": BO_MAX_NUM,
                "minebed/bo_objective_evals": bo_objective_evals,
                "minebed/configured_total_sim_budget": configured_total_sim_budget,
                "minebed/bo_training_sim_samples": bo_training_sim_samples,
                "minebed/estimated_total_sim_samples": estimated_total_sim_samples,
                "minebed/effective_datasize": DATASIZE,
                "minebed/effective_batchsize": BATCHSIZE,
            })
        del lc2st

        objects = {
            'post_samples': post_samples_1,
        }
        with open(f"{self.subdir}/posts_bed_obj.pkl", "wb") as f:
            pkl.dump(objects, f)
            bed_obj.save(f"{self.subdir}/bed_obj")


    def _init_logging(self):
        path = os.path.join(self.subdir, 'log.csv')
        logf = open(path, 'a')
        fieldnames = ['seed', 'EIG', 'opt_design', 'inference_time']
        writer = csv.DictWriter(logf, fieldnames=fieldnames)
        if os.stat(path).st_size == 0:
            writer.writeheader()
            logf.flush()
        return logf, writer


from BMP_minebed import Workspace as W

@hydra.main(version_base=None, config_path=".", config_name="bmp_minebed")
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
