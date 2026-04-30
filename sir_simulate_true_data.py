import os
import argparse
import sys
import time

import torch
import torchsde

from lfiax.utils.torch_utils import solve_sir_sdes

# needed for torchsde
sys.setrecursionlimit(1500)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Epidemic: solve SIR equations")
    # parser.add_argument("--num-samples", default=100000, type=int)
    # parser.add_argument("--device", default="cuda", type=str)

    if not os.path.exists("sed_data"):
        os.makedirs("sed_data")

    # args = parser.parse_args()
    num_samples = 3 # 100000
    device = 'cpu'
    
    # Convert your lists to torch tensors
    true_betas = torch.tensor([0.195, 0.3, 0.8])
    true_gammas = torch.tensor([0.15, 0.1, 0.1])

    # Stack the tensors along a new dimension
    combined_tensor = torch.stack((true_betas, true_gammas), dim=1) 

    for pair in combined_tensor:
        repeated_pair = pair.unsqueeze(0).repeat(num_samples, 1)
        R = pair[0] / pair[1]
        R_str = "{:.1f}".format(R).replace('.', '_') if '.' in "{:.1f}".format(R) else "{:.0f}".format(R)

        print("Generating initial training data...")
        solve_sir_sdes(
            num_samples=num_samples,
            device=device,
            grid=10000,
            save=True,
            savegrad=False,
            filename=f"sir_sde_data_{R_str}.pt",
            params=repeated_pair,
        )
        print("Generating initial test data...")
    ####### generate a big test dataset
    # test_data = []
    # for i in range(3):
    #     dict_i = solve_sir_sdes(
    #         num_samples=args.num_samples,
    #         device=args.device,
    #         grid=10000,
    #         save=False,
    #         savegrad=False,
    #     )
    #     test_data.append(dict_i)

    # save_dict = {
    #     "prior_samples": torch.cat([d["prior_samples"] for d in test_data]),
    #     "ys": torch.cat([d["ys"] for d in test_data], dim=1),
    #     "dt": test_data[0]["dt"],
    #     "ts": test_data[0]["ts"],
    #     "N": test_data[0]["N"],
    #     "I0": test_data[0]["I0"],
    # }
    # save_dict["num_samples"] = save_dict["prior_samples"].shape[0]
    # torch.save(save_dict, "data/sir_sde_data_test.pt")
