
from __future__ import absolute_import, division, print_function

import argparse, math
import timeit, time, os
import numpy as np
import torch
import torch.optim as optim
import sys
sys.path.append(os.getcwd())
from Data_Path import get_data_path
from Utils.data_gen import Task_Generator
from Utils.common import save_model_state, load_model_state, create_result_dir, set_random_seed, write_to_log, save_run_data, net_weights_magnitude, net_weights_diff
from Models.stochastic_models import get_model
from PriorMetaLearning import meta_test_Bayes, meta_train_Bayes_finite_tasks, meta_train_Bayes_infinite_tasks
from PriorMetaLearning.Analyze_Prior import run_prior_analysis

torch.backends.cudnn.benchmark = True  # For speed improvement with models with fixed-length inputs
# torch.set_default_tensor_type(torch.DoubleTensor) # For Assertion bug
# -------------------------------------------------------------------------------------------
#  Set Parameters
# -------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()

# ----- Run Parameters ---------------------------------------------#

parser.add_argument('--run-name', type=str, help='Name of dir to save results in (if empty, name by time)',
                    default='')

parser.add_argument('--gpu_index', type=int,
                    help='The index of GPU device to run on',
                    default=0)

parser.add_argument('--seed', type=int,  help='random seed',
                    default=1)

parser.add_argument('--mode', type=str, help='MetaTrain or LoadMetaModel',
                    default='MetaTrain')   # 'MetaTrain'  \ 'LoadMetaModel'

parser.add_argument('--load_model_path', type=str, help='set the path to pre-trained model, in case it is loaded (if empty - set according to run_name)',
                    default='')

parser.add_argument('--test-batch-size',type=int,  help='input batch size for testing (reduce if memory is limited)',
                    default=512)

parser.add_argument('--n_test_tasks', type=int,
                    help='Number of meta-test tasks for meta-evaluation (how many tasks to average in final result)',
                    default=20)


# ----- Task Parameters ---------------------------------------------#

parser.add_argument('--data-source', type=str, help="Data: 'MNIST' / 'CIFAR10' / Omniglot / SmallImageNet",
                    default='MNIST')

parser.add_argument('--n_train_tasks', type=int, help='Number of meta-training tasks (0 = infinite)',
                    default=10)

parser.add_argument('--data-transform', type=str, help="Data transformation:  'None' / 'Permute_Pixels' / 'Permute_Labels'/ Shuffled_Pixels ",
                    default='None')

parser.add_argument('--n_pixels_shuffles', type=int, help='In case of "Shuffled_Pixels": how many pixels swaps',
                    default=200)

parser.add_argument('--limit_train_samples_in_train_tasks', type=int,
                    help='Upper limit for the number of training samples in the meta-train tasks (0 = unlimited)',
                    default=600)

parser.add_argument('--limit_train_samples_in_test_tasks', type=int,
                    help='Upper limit for the number of training samples in the meta-test tasks (0 = unlimited)',
                    default=100)

# N-Way K-Shot Parameters:
parser.add_argument('--N_Way', type=int, help='Number of classes in a task (for Omniglot)',
                    default=5)

parser.add_argument('--K_Shot_MetaTrain', type=int,
                    help='Number of training sample per class in meta-training in N-Way K-Shot data sets',
                    default=100)  # Note:  test samples are the rest of the data

parser.add_argument('--K_Shot_MetaTest', type=int,
                    help='Number of training sample per class in meta-testing in N-Way K-Shot data sets',
                    default=100)  # Note:  test samples are the rest of the data

# SmallImageNet Parameters:
parser.add_argument('--n_meta_train_classes', type=int,
                    help='For SmallImageNet: how many categories are available for meta-training',
                    default=500)

# Omniglot Parameters:
parser.add_argument('--chars_split_type', type=str,
                    help='how to split the Omniglot characters  - "random" / "predefined_split"',
                    default='random')

parser.add_argument('--n_meta_train_chars'
                    , type=int, help='For Omniglot: how many characters to use for meta-training, if split type is random',
                    default=1200)

# ----- Algorithm Parameters ---------------------------------------------#

parser.add_argument('--complexity_type', type=str,
                    help=" The learning objective complexity type",
                    default='LLA')  #  'NoComplexity' /  'Variational_Bayes' / 'PAC_Bayes_Pentina'   McAllester / Seeger'"

# parser.add_argument('--override_eps_std', type=float,
#                     help='For debug: set the STD of epsilon variable for re-parametrization trick (default=1.0)',
#                     default=1.0)

parser.add_argument('--loss-type', type=str, help="Loss function",
                    default='CrossEntropy') #  'CrossEntropy' / 'L2_SVM'

parser.add_argument('--model-name', type=str, help="Define model type (hypothesis class)'",
                    default='ConvNet3')  # OmConvNet / 'FcNet3' / 'ConvNet3'

parser.add_argument('--batch-size', type=int, help='input batch size for training',
                    default=128)

parser.add_argument('--n_meta_train_epochs', type=int, help='number of epochs to train',
                    default=100)  # 150

parser.add_argument('--n_inner_steps', type=int,
                    help='For infinite tasks case, number of steps for training per meta-batch of tasks',
                    default=50)  #

parser.add_argument('--n_meta_test_epochs', type=int, help='number of epochs to train',
                    default=100)  #

parser.add_argument('--lr', type=float, help='initial learning rate',
                    default=1e-3)

parser.add_argument('--meta_batch_size', type=int, help='Maximal number of tasks in each meta-batch',
                    default=5)


parser.add_argument('--init_from_prior', default=True, type=lambda x: (str(x).lower() == 'true'))


# -------------------------------------------------------------------------------------------
#  More parameters
# -------------------------------------------------------------------------------------------

prm = parser.parse_args()
prm.device = torch.device("cuda:"+str(prm.gpu_index) if torch.cuda.is_available() else "cpu")
prm.data_path = get_data_path()


# Weights initialization (for Bayesian net):
prm.log_var_init = {'mean': -10, 'std': 0.1}  # The initial value for the log-var parameter (rho) of each weight

# Number of Monte-Carlo iterations:
prm.n_MC = 1

#  Define optimizer:
prm.optim_func, prm.optim_args = optim.Adam,  {'lr': prm.lr}  #'weight_decay': 1e-4
# prm.optim_func, prm.optim_args = optim.Adam,  {'lr': prm.lr, 'amsgrad': True}  #'weight_decay': 1e-4
# prm.optim_func, prm.optim_args = optim.SGD, {'lr': prm.lr, 'momentum': 0.9}

# Learning rate decay schedule:
# prm.lr_schedule = {'decay_factor': 0.1, 'decay_epochs': [50, 150]}
prm.lr_schedule = {}  # No decay

# MPB alg  params:
prm.kappa_prior = 1e2  #  parameter of the hyper-prior regularization
prm.kappa_post = 1e-3  # The STD of the 'noise' added to prior
prm.delta = 0.1  #  maximal probability that the bound does not hold

# prm.test_batch_size = 128

init_from_prior = prm.init_from_prior  #  False \ True . In meta-testing -  init posterior from learned prior

# Test type:
prm.test_type = 'MaxPosterior' # 'MaxPosterior' / 'MajorityVote' / 'AvgVote'


# -------------------------------------------------------------------------------------------
#  Init run
# -------------------------------------------------------------------------------------------

create_result_dir(prm)

set_random_seed(prm.seed)


# path to save the learned meta-parameters for both models (prior and initiation)
save_path_a = os.path.join(prm.result_dir, 'model_a.pt')
save_path_p = os.path.join(prm.result_dir, 'model_p.pt')

task_generator = Task_Generator(prm)

# -------------------------------------------------------------------------------------------
#  Run Meta-Training
# -------------------------------------------------------------------------------------------

start_time = timeit.default_timer()

if prm.mode == 'MetaTrain':
    n_train_tasks = prm.n_train_tasks
    limit_train_samples_in_train_tasks = prm.limit_train_samples_in_train_tasks
    if n_train_tasks:
        # In this case we generate a finite set of train (observed) task before meta-training.
        # Generate the data sets of the training tasks:
        write_to_log('--- Generating {} training-tasks'.format(n_train_tasks), prm)
        train_data_loaders = task_generator.create_meta_batch(prm, n_train_tasks, limit_train_samples=limit_train_samples_in_train_tasks, meta_split='meta_train')

        # Meta-training to learn prior:
        prior_model_p, prior_model_a = meta_train_Bayes_finite_tasks.run_meta_learning(train_data_loaders, prm)
        # save learned priors:
        save_model_state(prior_model_a, save_path_a)
        save_model_state(prior_model_p, save_path_p)
        write_to_log('Trained prior saved in ' + save_path_a, prm)
    else:
        # In this case we observe new tasks generated from the task-distribution in each meta-iteration.
        write_to_log('---- Infinite train tasks - New training tasks are '
                     'drawn from tasks distribution in each iteration...', prm)

        # Meta-training to learn meta-prior (theta params):
        prior_model = meta_train_Bayes_infinite_tasks.run_meta_learning(task_generator, prm)


elif prm.mode == 'LoadMetaModel':

    # Loads  previously training prior.
    # First, create the model:
    prior_model_p = get_model(prm)
    prior_model_a = get_model(prm)
    # Then load the weights:
    load_model_state(prior_model_a, prm.load_model_path)
    load_model_state(prior_model_p, prm.load_model_path)
    write_to_log('Pre-trained priors loaded from ' + prm.load_model_path, prm)
else:
    raise ValueError('Invalid mode')

# Some information about prior models (for prior and initiation)
total_params = sum(p.numel() for p in prior_model_a.parameters())
norm_sqr_a = net_weights_magnitude(prior_model_a, prm, p=2)
norm_sqr_p = net_weights_magnitude(prior_model_p, prm, p=2)
diff_sqr = net_weights_diff(prior_model_a, prior_model_p, prm, p=2)

# -------------------------------------------------------------------------------------------
# Generate the data sets of the test tasks:
# -------------------------------------------------------------------------------------------

n_test_tasks = prm.n_test_tasks

limit_train_samples_in_test_tasks = prm.limit_train_samples_in_test_tasks
if limit_train_samples_in_test_tasks == 0:
    limit_train_samples_in_test_tasks = None

write_to_log('---- Generating {} test-tasks with at most {} training samples'.
             format(n_test_tasks, limit_train_samples_in_test_tasks), prm)


test_tasks_data = task_generator.create_meta_batch(prm, n_test_tasks, meta_split='meta_test',
                                                   limit_train_samples=limit_train_samples_in_test_tasks)
#
# -------------------------------------------------------------------------------------------
#  Run Meta-Testing
# -------------------------------------------------------------------------------
write_to_log('Meta-Testing with transferred prior....', prm)

test_err_vec = np.zeros(n_test_tasks)
for i_task in range(n_test_tasks):
    print('Meta-Testing task {} out of {}...'.format(1+i_task, n_test_tasks))
    task_data = test_tasks_data[i_task]
    test_err_vec[i_task], _ = meta_test_Bayes.run_learning(task_data, prior_model_p, prior_model_a, prm, init_from_prior, verbose=0)


# save result
save_run_data(prm, {'test_err_vec': test_err_vec})

# -------------------------------------------------------------------------------------------
#  Print results
# -------------------------------------------------------------------------------------------
#  Print prior analysis
run_prior_analysis(prior_model_p)

stop_time = timeit.default_timer()
write_to_log('Total runtime: ' +
             time.strftime("%H hours, %M minutes and %S seconds", time.gmtime(stop_time - start_time)),  prm)

#  Print results
write_to_log('----- Final Results: ', prm)
write_to_log('----- Meta-Testing - Avg test err: {:.3}%, STD: {:.3}%'
             .format(100 * test_err_vec.mean(), 100 * test_err_vec.std()), prm)

print('Prior Model A Info:', math.sqrt(norm_sqr_a), '\tPrior Model P Info: ', math.sqrt(norm_sqr_p))
print('Prior Models Diff:', math.sqrt(diff_sqr), '\tTotal params: ', total_params, '\n\n')