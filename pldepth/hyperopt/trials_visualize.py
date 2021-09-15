from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK
from hyperopt.pyll import scope as ho_scope
from hyperopt.pyll.stochastic import sample as ho_sample
import pickle
from pldepth.hyperopt.hyperparams import lr_dict
from matplotlib import pyplot as plt
import numpy as np

class HyperoptAnalyser():
    def __init__(self, trial):
        self.trials = trial
        res_list = trial.results
        self.loss_vec = [x['loss'] for x in res_list]

    def extract_trials(self, par):
        """
        returns the chosen hyperparameters and their corresponding loss as 2 vectors
        """
        return self.trials.vals[par], self.loss_vec

    def plot_param(self, par):
        """
        plots the hyperparameter vs loss plot for the given parameter
        """
        x , y = self.extract_trials(par)
        if par=='lr':
            plt.plot(np.log(x),y, 'o')
        else:
            plt.plot(x, y, 'o')
        plt.xlabel(par)
        plt.ylabel('loss')
        plt.show()


    def get_best_params(self):
        return self.trials.best_trial['result']['loss'], self.trials.best_trial['misc']['vals']

    def get_params_names(self):
        return list(self.trials.vals.keys())



if __name__ == "__main__":
    trials = pickle.load(open("/home/praneeth/projects/thesis/git/PLDepth/trials/activ_on_info_130921-125645", "rb"))
    #samples = f_unpack_dict(f_wrap_space_eval(lr_dict, trials))
    a = HyperoptAnalyser(trials)
    #
    pars = a.get_params_names()
    print(pars)
    print(a.get_best_params())
    a.plot_param('rpi')

