import argparse
import json
import os

from nas_model import NASModel

from smac.facade.smac_facade import SMAC
from smac.scenario.scenario import Scenario
from smac.tae.execute_func import ExecuteTAFuncDict

"""
This is based on the nas_benchmarks implementation
https://github.com/automl/nas_benchmarks/blob/master/experiment_scripts/run_smac.py
"""

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', default="public_data/devel_dataset_0", type=str, nargs='?', help='path to the dataset')
parser.add_argument('--max_epochs', default=64, type=int, nargs='?', help='max num training epochs')
parser.add_argument('--run_id', default=0, type=int, nargs='?', help='unique number to identify this run')
parser.add_argument('--n_iters', default=100, type=int, nargs='?', help='number of iterations for optimization method')
parser.add_argument('--output_path', default="./", type=str, nargs='?',
                    help='specifies the path where the results will be saved')
parser.add_argument('--n_trees', default=10, type=int, nargs='?', help='number of trees for the random forest')
parser.add_argument('--random_fraction', default=.33, type=float, nargs='?', help='fraction of random configurations')
parser.add_argument('--max_feval', default=4, type=int, nargs='?',
                    help='maximum number of function evaluation per configuration')

args = parser.parse_args()

b = NASModel(dataset_path=args.dataset_path, 
             max_epochs=args.max_epochs, 
             multi_fidelity=False)

output_path = os.path.join(args.output_path, "smac")
os.makedirs(os.path.join(output_path), exist_ok=True)

cs = b.get_configuration_space()

scenario = Scenario({"run_obj": "quality",
                     "runcount-limit": args.n_iters,
                     "wallclock-limit": 86300,
                     "cs": cs,
                     "deterministic": "false",
                     "initial_incumbent": "RANDOM",
                     "output_dir": ""})


def objective_function(config, **kwargs):
    y, c = b.objective_function(config)
    return float(y)


tae = ExecuteTAFuncDict(objective_function, use_pynisher=False)
smac = SMAC(scenario=scenario, tae_runner=tae, run_id=args.run_id)

# probability for random configurations

smac.solver.random_configuration_chooser.prob = args.random_fraction
smac.solver.model.rf_opts.num_trees = args.n_trees

# only 1 configuration per SMBO iteration
smac.solver.scenario.intensification_percentage = 1e-10
smac.solver.intensifier.min_chall = 1

# maximum number of function evaluations per configuration
smac.solver.intensifier.maxR = args.max_feval

smac.optimize()

res = b.get_results(ignore_invalid_configs=True)

fh = open(os.path.join(output_path, 'run_%d.json' % args.run_id), 'w')
json.dump(res, fh)
fh.close()
