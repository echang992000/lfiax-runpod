"""Simulator information and sampling."""

SIMULATORS = {
    "bmp": {
        "name": "bmp",
        "full_name": "Bone Morphogenetic Protein",
        "description": "ODE model of the BMP signalling pathway",
        "num_params": 4,
        "parameters": [
            {"name": "k1", "description": "Rate constant 1", "range": [0.01, 10.0]},
            {"name": "k2", "description": "Rate constant 2", "range": [0.01, 10.0]},
            {"name": "k3", "description": "Rate constant 3", "range": [0.01, 10.0]},
            {"name": "k4", "description": "Rate constant 4", "range": [0.01, 10.0]},
        ],
        "design_space": "Concentration levels (continuous)",
        "output_dim": 1,
        "module": "lfiax.utils.simulators",
    },
    "sir": {
        "name": "sir",
        "full_name": "SIR Epidemic Model",
        "description": "Stochastic compartmental epidemic model (Susceptible-Infected-Recovered)",
        "num_params": 2,
        "parameters": [
            {"name": "beta", "description": "Infection rate", "range": [0.01, 1.0]},
            {"name": "gamma", "description": "Recovery rate", "range": [0.01, 1.0]},
        ],
        "design_space": "Observation time points",
        "output_dim": 3,
        "module": "lfiax.utils.simulators",
    },
    "linear_regression": {
        "name": "linear_regression",
        "full_name": "Linear Regression",
        "description": "Closed-form toy problem for smoke tests",
        "num_params": 2,
        "parameters": [
            {"name": "slope", "description": "Regression slope", "range": [-5.0, 5.0]},
            {"name": "intercept", "description": "Regression intercept", "range": [-5.0, 5.0]},
        ],
        "design_space": "Input x values",
        "output_dim": 1,
        "module": "lfiax.utils.simulators",
    },
    "two_moons": {
        "name": "two_moons",
        "full_name": "Two Moons",
        "description": "Standard SBI benchmark with crescent-shaped posteriors",
        "num_params": 2,
        "parameters": [
            {"name": "theta_1", "description": "Parameter 1", "range": [-1.0, 1.0]},
            {"name": "theta_2", "description": "Parameter 2", "range": [-1.0, 1.0]},
        ],
        "design_space": "None (fixed design)",
        "output_dim": 2,
        "module": "sbibm.tasks",
    },
}


def list_simulators():
    """Return info about all available simulators.

    Returns:
        list[dict]: Simulator summaries.
    """
    return [
        {
            "name": s["name"],
            "full_name": s["full_name"],
            "description": s["description"],
            "num_params": s["num_params"],
        }
        for s in SIMULATORS.values()
    ]


def get_simulator_info(name):
    """Get detailed info about a specific simulator.

    Args:
        name: Simulator name.

    Returns:
        dict: Detailed simulator info.

    Raises:
        ValueError: If simulator not found.
    """
    if name not in SIMULATORS:
        raise ValueError(
            f"Unknown simulator '{name}'. "
            f"Available: {', '.join(SIMULATORS.keys())}"
        )
    return SIMULATORS[name]


def sample_prior(simulator_name, n_samples=10, seed=None):
    """Sample from the prior distribution of a simulator.

    Args:
        simulator_name: Name of the simulator.
        n_samples: Number of samples to draw.
        seed: Random seed.

    Returns:
        dict: {simulator, n_samples, samples: list[list[float]]}
    """
    info = get_simulator_info(simulator_name)

    try:
        import jax
        import jax.numpy as jnp
        import jax.random as jr

        key = jr.PRNGKey(seed if seed is not None else 0)

        # Simple uniform prior sampling based on parameter ranges
        samples = []
        for _ in range(n_samples):
            key, subkey = jr.split(key)
            sample = []
            for param in info["parameters"]:
                lo, hi = param["range"]
                val = jr.uniform(subkey, shape=(), minval=lo, maxval=hi)
                sample.append(float(val))
                key, subkey = jr.split(key)
            samples.append(sample)

        return {
            "simulator": simulator_name,
            "n_samples": n_samples,
            "param_names": [p["name"] for p in info["parameters"]],
            "samples": samples,
        }
    except ImportError:
        return {
            "simulator": simulator_name,
            "error": "JAX not available. Install JAX to sample from simulators.",
        }


def simulate(simulator_name, theta, design=None, seed=None):
    """Run a forward simulation.

    Args:
        simulator_name: Name of the simulator.
        theta: Parameter values (list of floats).
        design: Optional design values.
        seed: Random seed.

    Returns:
        dict: Simulation result.
    """
    info = get_simulator_info(simulator_name)

    if len(theta) != info["num_params"]:
        raise ValueError(
            f"Expected {info['num_params']} parameters for '{simulator_name}', "
            f"got {len(theta)}"
        )

    return {
        "simulator": simulator_name,
        "theta": theta,
        "design": design,
        "seed": seed,
        "note": "Full simulation requires lfiax + JAX. Use 'experiment run' for complete workflows.",
    }
