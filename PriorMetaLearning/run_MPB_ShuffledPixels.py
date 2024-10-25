from subprocess import call
import argparse
import sys, os
sys.path.append(os.getcwd())
n_train_tasks = 10

parser = argparse.ArgumentParser()

parser.add_argument('--n_shuffles', type=int, help='no of shuffled pixels', default=200)
parser.add_argument('--comp', type=str,
                    help="The learning objective complexity type",
                    default='LLA')
# 'NoComplexity' / 'LLA' / 'Variational_Bayes' / 'PAC_Bayes_Pentina' /'McAllester / 'Seeger'
parser.add_argument('--seed', type=int,  help='random seed',
                    default=1)
parser.add_argument('--gpu_id', type=int,
                    help='The index of GPU device to run on',
                    default=0)
parser.add_argument('--test_tasks', type=int,
                    help='Number of meta-test tasks for meta-evaluation (how many tasks to average in final result)',
                    default=20)
parser.add_argument('--train_epochs', type=int, help='number of epochs to train',
                    default=100) 
parser.add_argument('--test_epochs', type=int, help='number of epochs to train',
                    default=100)
parser.add_argument('--train_samples', type=int,
                    help='Upper limit for the number of training samples in the meta-train tasks (0 = unlimited)',
                    default=600)
parser.add_argument('--test_samples', type=int,
                    help='Upper limit for the number of training samples in the meta-test tasks (0 = unlimited)',
                    default=100)

args = parser.parse_args()

n_shuffles = args.n_shuffles
comp = args.comp
seed = args.seed
gpu_id = args.gpu_id
test_tasks = args.test_tasks
train_epochs = args.train_epochs
test_epochs = args.test_epochs
train_samples = args.train_samples
test_samples = args.test_samples

call(['python', 'PriorMetaLearning/main_Meta_Bayes.py',
      '--run-name', 'Shuffled_{}_Pixels_{}_TrTask_{}_TeTask_{}_TrSample_{}_TeSample_{}_TrEpoch_{}_TeEpoch_{}_Comp_{}_Seed'.format(
          n_shuffles, n_train_tasks, test_tasks, train_samples, test_samples,
          train_epochs, test_epochs, comp, seed),
      '--gpu_index', str(gpu_id),
      '--data-source', 'MNIST',
      '--data-transform', 'Shuffled_Pixels',
      '--n_pixels_shuffles', str(n_shuffles),
      '--limit_train_samples_in_train_tasks', str(train_samples),
      '--limit_train_samples_in_test_tasks', str(test_samples),
      '--n_train_tasks',  str(n_train_tasks),
      '--mode', 'MetaTrain',
      '--complexity_type',  comp,
      '--model-name', 'FcNet3',
      '--n_meta_train_epochs', str(train_epochs),
      '--n_meta_test_epochs', str(test_epochs),
      '--n_test_tasks', str(test_tasks),
      '--meta_batch_size', '5',
      '--seed', str(seed),
      ])
