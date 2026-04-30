from typing import Any, Callable, Dict, Optional, Union, AnyStr, Tuple
from xmlrpc.client import boolean
import numpy as np
import random, os, sys
import importlib
import multiprocessing
import functools
import itertools
import jax.numpy as jnp
from tqdm import tqdm as tqdm

# from .promisys import bmp_util as psb
from bmp_simulator.promisys import bmp_util as psb

Array = jnp.ndarray


def bmp_simulator(
    d: np.ndarray,
    p: np.ndarray,
    model_size: Tuple[int, int, int] = (1, 1, 1),
    model: str = "onestep",
    fixed_receptor: bool = True,
    n_threads: int = 4,
):
    """
    Purpose is to wrap the promisys simulator and return a numpy
    array compatible with minebed and sbi. Need to split the processing of the
    inputs depending on whether fixed_receptor=True.

    Args:
        d: array_like, shape (num_designs,) or (1,) Design (of ligands) that will be put into the simulator. 
            Assign as class attribute for promisys. 
        p: array_like, shape (2 * n_L * n_A * n_B,) for onestep or 
            ((2 * n_L * n_A * n_B) + n_L * n_A,) for twostep or (len(priors), ) for a custom simulator.
            Numpy array sampled from the distribution object of the priors, either as a 
            Design (of ligands) that will be put into the simulator.
            Assign as class attribute for promisys.
        p: array_like, shape (2 * n_L * n_A * n_B,) for onestep or ((2 * n_L * n_A * n_B) + n_L * n_A,) for twostep or (len(priors), ) for a custom simulator.
            Numpy array sampled from the distribution object of the priors, either as a
            torch distribution or sbi posterior object.
        model_size: tuple of ints, shape (3,) BMP model to simulate for number of unique ligands, type 1, and type 2 receptors.
        mode: Whether to use the 'onestep' or 'twostep' BMP model
        fixed_receptor: Determine whether to use bmp model with known/fixed receptor, or stochastic according to log distribution. (see promisys code)

    Returns:
        S: something
    """
    # Check that the design is the right size (N, 1) not (1, N)
    n_L, n_A, n_B = model_size
    num_receptors = n_A + n_B
    ligands = d.T
    
    # Splitting passed prior's columns to work with promisys
    if fixed_receptor:
        if model == "onestep":
            # assert that p passed is correct size for constant receptor
            if p.shape[1] != (n_L * n_A * n_B + n_L * n_A * n_B):
                raise ValueError(
                    "Number of priors is not consistent with model size or fixed_receptor value."
                )

            p = np.hsplit(p, [n_L * n_A * n_B])

        elif model == "twostep":
            if p.shape[1] != (n_L * n_A * n_B + n_L * n_A + n_L * n_A * n_B):
                raise ValueError(
                    "Number of priors is not consistent with model size or fixed_receptor value."
                )

            p = np.hsplit(p, [n_L * n_A * n_B + n_L * n_A])

        # ----- multiprocessing starmap ------
        # Multiprocess the simulations
        Rs = None
        n_threads = n_threads
        
        sim_values = []
        for d in ligands:
            iteration_sim_values = []
            for params in zip(*p):
                S = psb.sim_S_LAB(model_size, d, Rs, model=model, 
                                fixed_receptor=fixed_receptor, *params)
                iteration_sim_values.append(S)
            sim_values.append(iteration_sim_values)
        inner_concatenated = [jnp.vstack(inner_list) for inner_list in sim_values]
        final_array = jnp.concatenate(inner_concatenated, axis=1)

    else:
        if model == "onestep":
            # assert that p passed is correct size for constant receptor
            if p.shape[1] != (n_A + n_B + n_L * n_A * n_B + n_L * n_A * n_B):
                raise ValueError(
                    "Number of priors is not consistent with model size or fixed_receptor value."
                )

            p = np.hsplit(p, [num_receptors, num_receptors + n_L * n_A * n_B])

        elif model == "twostep":
            if p.shape[1] != (
                n_A + n_B + n_L * n_A * n_B + n_L * n_A + n_L * n_A * n_B
            ):
                raise ValueError(
                    "Number of priors is not consistent with model size or fixed_receptor value."
                )

            p = np.hsplit(
                p, [num_receptors, num_receptors + n_L * n_A * n_B + n_L * n_A]
            )

        # ----- multiprocessing starmap ------
        # Multiprocess the simulations
        n_threads = n_threads
        with multiprocessing.Pool(n_threads) as pools:
            # %time
            S = pools.starmap(
                functools.partial(
                    psb.sim_S_LAB,
                    model_size,
                    ligands,
                    fixed_receptor=fixed_receptor,
                    model=model,
                ),
                zip(*p),
            )

    # Turn promisys output into numpy arrays
    S = np.array(final_array)

    # Temporary fix for nL=1 condition & shapes with extra dimensions
    if len(S.shape) > 2:
        S = S.squeeze(axis=-1)

    return S

def simulate_bmp_data_Ls(
        L: Array,
        k_prior: Array,
        e_prior: Array):
    """
    Returns the data from the simulator. Need to have unnormalized L values.
    Uses fixed_receptor=True, so no receptor values needed.
    """
    iteration_sim_values = []
    chunk_size = 1  # Process 1 sample at a time

    # Make sure all parameter arrays have the same length.
    n_samples = len(k_prior)

    # Loop over the parameter arrays in chunks.
    for i in range(0, n_samples, chunk_size):
        # Slice out the current chunk for each parameter.
        P_chunk = (
            k_prior[i:i + chunk_size],
            e_prior[i:i + chunk_size]
        )
        # Also slice the corresponding ligand concentrations.
        L_chunk = L[i:i + chunk_size]
        
        # For each tuple of parameters in the current chunk, run the simulation.
        for params in zip(*P_chunk):
            # Fix: Rs should be positional argument, not keyword
            S = psb.sim_S_LAB((1, 1, 1), L_chunk, None, model="onestep",
                            fixed_receptor=True, *params)
            iteration_sim_values.append(S)

    # Combine all simulation results
    x = jnp.hstack(iteration_sim_values)
    return x

def simulate_bmp_experiment_conditions(Ls, theta_0):
    """
    Simulate BMP data using provided Ls slices with fixed receptors.
    
    Args:
        Ls: Array of L samples, shape (1, num_samples)
        theta_0: Array containing the theta samples, shape (num_samples, 2)
    
    Returns:
        x: Simulated data with shape (num_samples, 1)
    """
    num_samples = theta_0.shape[0]
    all_sim_results = []
    
    # Iterate over samples: each L[i] goes with theta_0[i, :]
    for i in range(num_samples):
        # Get the i-th L value and i-th theta
        L_sample = Ls[0, i:i+1]  # Shape (1,)
        theta_sample = theta_0[i:i+1, :]  # Shape (1, 2)
        
        # Call simulation function
        result_i = simulate_bmp_data_Ls(L_sample, theta_sample[:, :1], theta_sample[:, 1:])
        all_sim_results.append(result_i)
    
    # Stack the results
    x = jnp.vstack(all_sim_results)
    return x.T.squeeze(0)
