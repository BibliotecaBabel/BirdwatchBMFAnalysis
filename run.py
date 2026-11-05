# run.py
import hydra
import os

import numpy as np

from omegaconf import OmegaConf

from bmf.experiment import Experiment
from bmf.config import ExperimentConfig

# this allows using numpy for specifying parameters
OmegaConf.register_new_resolver(
    "linspace", 
    lambda start, stop, num: ','.join(str(x) for x in np.linspace(start, stop, num))
)

@hydra.main(version_base="1.3", config_path="conf", config_name="base")
def main(cfg: ExperimentConfig):    
    # Run experiment
    exp = Experiment(cfg)
    result = exp.run_experiment(
        comet_tags=['hydra'] + (['debug'] if cfg.debug_mode else [])
    )
    
    print(f"Experiment completed: {result['name']}")

if __name__ == "__main__":
    main()