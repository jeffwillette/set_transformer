from hpbandster.core.worker import Worker
import pickle
from consistent import ConsistentModel
import time
import json
import time
import logging
import argparse
import numpy as np
import ConfigSpace as CS
from hpbandster.optimizers import BOHB
import hpbandster.core.nameserver as hpns
import os
import torch
from mvn_diag import MultivariateNormalDiag
from mixture_of_mvns import MixtureOfMVNs


def get_results_path(args):
    return args.savedir

def generate_benchmark():
    if not os.path.isdir('benchmark'):
        os.makedirs('benchmark')
    N_list = np.random.randint(N_min, N_max, args.num_bench)
    data = []
    ll = 0.
    for N in tqdm(N_list):
        X, labels, pi, params = mog.sample(B, N, K, return_gt=True)
        ll += mog.log_prob(X, pi, params).item()
        data.append(X)
    bench = [data, ll/args.num_bench]
    torch.save(bench, benchfile)


class MinibatchTrainer(object):
    def __init__(self, args):
        self.args = args

    def init_model(self, config):
        self.config = config

        self.model = ConsistentModel(
            in_dim=config["in_dim"],
            out_dim=config["out_dim"],
            num_outputs=config["num_outputs"],
            hidden_dim=config["hidden_dim"],
            extractor=config["extractor"],
            K=[config["K"]],
            h=[config["h"]],
            d=[config["d"]],
            d_hat=[config["d"]],
            _slots=config["_slots"],
            g=config["g"],
            ln=config["ln"]
        ).cuda()

        self.opt = torch.optim.Adam(self.model.parameters())
        self.mvn = MultivariateNormalDiag(2)
        self.mog = MixtureOfMVNs(self.mvn)
        self.bench = torch.load(self.args.benchfile)

    def fit(self, epochs):
        self.train(epochs)

    def save_checkpoint(self, epoch: int) -> None:
        checkpoint = {"model": self.model.state_dict(), "epoch": epoch}
        torch.save(checkpoint, os.path.join(get_results_path(self.args), "checkpoint.pt"))

    def load_checkpoint(self) -> None:
        checkpoint = torch.load(os.path.join(get_results_path(self.args), "checkpoint.pt"))
        self.model.load_state_dict(checkpoint["model"])

    def generate_benchmark(self):
        if not os.path.isdir('benchmark'):
            os.makedirs('benchmark')

        N_list = np.random.randint(N_min, N_max, args.num_bench)
        data = []
        ll = 0.
        for N in tqdm(N_list):
            X, labels, pi, params = self.mog.sample(args.B, N, args.K, return_gt=True)
            ll += self.mog.log_prob(X, pi, params).item()
            data.append(X)

        self.bench = [data, ll/args.num_bench]
        torch.save(self.bench, self.args.benchfile)

    def train(self, epochs):
        tick = time.time()
        for t in range(1, int(epochs)+1):
            if t == int(0.5*int(epochs)):
                self.opt.param_groups[0]['lr'] *= 0.1

            self.model.train()
            self.opt.zero_grad()
            N = np.random.randint(args.N_min, args.N_max)
            X = self.mog.sample(args.B, N, args.K)
            ll = self.mog.log_prob(X, *self.mvn.parse(self.model(X)))
            loss = -ll 
            loss.backward()
            self.opt.step()

        torch.save({'state_dict':self.model.state_dict()},
            os.path.join(self.args.savedir, 'model.tar'))

    def test(self, verbose=True):
        self.model.eval()
        data, oracle_ll = self.bench
        avg_ll = 0.
        for X in data:
            X = X.cuda()
            avg_ll += self.mog.log_prob(X, *self.mvn.parse(self.model(X))).item()
        avg_ll /= len(data)
        line = 'test ll {:.4f} (oracle {:.4f})'.format(avg_ll, oracle_ll)
        if verbose:
            logging.basicConfig(level=logging.INFO)
            gpu = os.environ["CUDA_VISIBLE_DEVICES"]
            logger = logging.getLogger(f'consistent-{gpu}')
            logger.addHandler(logging.FileHandler(
                os.path.join(self.args.savedir, 'test.log'), mode='w'))
            logger.info(line)
        return avg_ll, oracle_ll

    def plot(self):
        net.eval()
        X = self.mog.sample(B, np.random.randint(N_min, N_max), args.K)
        pi, params = self.mvn.parse(net(X))
        ll, labels = self.mog.log_prob(X, pi, params, return_labels=True)
        fig, axes = plt.subplots(2, B//2, figsize=(7*B//5,5))
        self.mog.plot(X, labels, params, axes)
        plt.show()

class BaseWorker(Worker):
    def __init__(self, *args, pargs, trainer, **kwargs):
        super().__init__(*args, **kwargs)

        self.args = pargs
        self.trainer = trainer
        self.results_path = get_results_path(self.args)

    def compute(self, config, budget, **kwargs):
        """compute is for baseline models"""

        self.trainer.init_model(config)
        self.trainer.fit(budget)

        # eval and return stats on both training and validation set
        avg_ll, oracle_ll = self.trainer.test()
        info = {"val": {"nll": avg_ll, "oracle": oracle_ll}}

        logger.info(f"val nll: {avg_ll}")
        return {"loss": -avg_ll, "info": info}

    def write_results(self, obj, res) -> None:
        with open(os.path.join(self.results_path, "tuned_settings.json"), "w") as outfile:
            json.dump(obj, outfile, indent=2)

        with open(os.path.join(self.results_path, "res.pkl"), "wb") as f:
            pickle.dump(res, f)

    @staticmethod
    def get_configspace(args):
        raise NotImplementedError("need to implement configspace for every model worker")


def load_config(args):
    path = args.savedir
    with open(os.path.join(path, "tuned_settings.json"), "r") as f:
        config = json.load(f)

    return config

def tune(args, worker, run_id, host, port, NS, log):
    worker.run(background=True)

    bohb = BOHB(
        configspace=worker.get_configspace(args),
        run_id=run_id,
        nameserver=host,
        nameserver_port=port,
        min_budget=args.min_epochs,
        max_budget=args.max_epochs,
        logger=log
    )

    log.info(f"starting: {run_id}")

    res = bohb.run(n_iterations=args.n_iterations)
    bohb.shutdown(shutdown_workers=True)
    NS.shutdown()

    id2config = res.get_id2config_mapping()
    incumbent = res.get_incumbent_id()
    runs = res.get_runs_by_id(incumbent)

    worker.write_results(
        {
            "best": id2config[incumbent]['config'],
            "total_sampled": len(id2config.keys()),
            "total_runs": len(res.get_all_runs()),
            "loss": runs[0].loss,
            "info": runs[0].info
        },
        res
    )


def train(args, trainer, log, save = True):
    config = load_config(args)["best"]
    log.info(f"training: {get_results_path(args)}")

    trainer.init_model(config)
    trainer.fit(args.max_epochs)
    if save:
        log.info("saving checkpoint")
        trainer.save_checkpoint(args.max_epochs)


def test(args, trainer, log):
    config = load_config(args)["best"]
    log.info(f"test: {get_results_path(args)}")

    trainer.init_model(config)
    trainer.load_checkpoint()
    return trainer.test()

class MinibatchWorker(BaseWorker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def get_configspace(args):
        cs = CS.ConfigurationSpace()

        cs.add_hyperparameter(CS.Constant('in_dim', value=2))
        cs.add_hyperparameter(CS.Constant('hidden_dim', value=128))
        cs.add_hyperparameter(CS.Constant('out_dim', value=4))
        cs.add_hyperparameter(CS.Constant('n_layers', value=4))
        cs.add_hyperparameter(CS.Constant('num_outputs', value=4))
        cs.add_hyperparameter(CS.Constant('d', value=128))
        cs.add_hyperparameter(CS.UniformIntegerHyperparameter('K', lower=1, upper=128))
        cs.add_hyperparameter(CS.UniformIntegerHyperparameter('h', lower=1, upper=128))
        cs.add_hyperparameter(CS.CategoricalHyperparameter('_slots', choices=['Learned', 'Random']))
        cs.add_hyperparameter(CS.CategoricalHyperparameter('g', choices=['max', 'min', 'mean', 'sum']))
        cs.add_hyperparameter(CS.CategoricalHyperparameter('extractor', choices=['mean', 'mean2', 'max', 'max2']))
        cs.add_hyperparameter(CS.CategoricalHyperparameter('ln', choices=[False, True]))

        return cs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, default="consistent", help="run id to give to the worker and nameserver")
    parser.add_argument("--shared_dir", type=str, default=".", help="shared dir to store runfiles")
    parser.add_argument("--nic_name", type=str, default="eth0", help="network interface name")
    parser.add_argument("--host", type=str, default='127.0.0.1', help="the host to run on (localhost)")
    parser.add_argument("--worker", action="store_true", help="whether or not this is a worker")
    parser.add_argument("--mode", type=str, default="tune", help="the training mode to run")

    # for the model
    parser.add_argument("--N_min", type=int, default=100, help="mimimum number to sample")
    parser.add_argument("--N_max", type=int, default=500, help="maximum number to sample")
    parser.add_argument("--B", type=int, default=10, help="batch (?)")
    parser.add_argument("--min_epochs", type=int, default=5000, help="minimum number of epochs to train")
    parser.add_argument("--max_epochs", type=int, default=50000, help="minimum number of epochs to train")
    parser.add_argument("--savedir", type=str, default="results/consistent", help="the directory to save the results in")
    parser.add_argument("--n_iterations", type=int, default=1, help="the number of BOHB iterations to run")
    parser.add_argument("--K", type=int, default=4, help="the number of clusters for the MOG")

    args = parser.parse_args()

    args.benchfile = "benchmark/mog_{:d}.pkl".format(args.K)
    if not os.path.isdir(args.savedir):
        os.makedirs(args.savedir)

    if not os.path.isfile(args.benchfile):
        generate_benchmark()

    gpu = os.environ["CUDA_VISIBLE_DEVICES"]
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(f'consistent-{gpu}')
    logger.addHandler(logging.FileHandler(
        os.path.join(args.savedir,
            'train_'+time.strftime('%Y%m%d-%H%M')+'.log'),
        mode='w'))
    logger.info(str(args) + '\n')

    trainer = MinibatchTrainer(args)
    if args.worker:
        time.sleep(5)
        w = MinibatchWorker(pargs=args, trainer=trainer, run_id=args.run_id, host=args.host, logger=logger)
        w.load_nameserver_credentials(working_directory=args.shared_dir)
        w.run(background=False)
        exit(0)

    NS = hpns.NameServer(run_id=args.run_id, host=args.host, port=0, working_directory=args.shared_dir)
    host, port = NS.start()

    worker = MinibatchWorker(pargs=args, trainer=trainer, run_id=args.run_id, nameserver=host, nameserver_port=port, logger=logger)
    worker.run(background=True)

    if args.mode == "tune":
        tune(args, worker, args.run_id, host, port, NS, logger)
    elif args.mode == "train":
        train(args, trainer, logger)
    elif args.mode == "test":
        test(args, trainer, logger)
