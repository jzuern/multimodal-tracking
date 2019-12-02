from __future__ import absolute_import
from SiamRPNEval import TrackerSiamRPNEval
from train.experimentrgbt import ExperimentRGBT


if __name__ == '__main__':

    experiment_dir = 'experiments/SiamRPN_RGBT-234_1'


    # If modality==1: use only RGB. If modality==2: RGB+IR
    modality = 1

    if modality == 1:
        net_path = '/home/zuern/siamRPN/train/experiments/default/model/model_e26.pth'
        # net_path = '/home/zuern/siamRPN/train/experiments/default/model/model_e1.pth'
        # net_path = experiment_dir + '/model/model_e1.pth'
        experiment_name = 'exp'
    else:
        net_path = experiment_dir + '/model/model_e1.pth'
        experiment_name = 'exp'



    tracker = TrackerSiamRPNEval(modality=modality,
                                 model_path=net_path)

    experiments = ExperimentRGBT('/home/zuern/datasets/thermal_tracking/RGB-T234/',
                       experiment_name=experiment_name,
                       experiment_dir=experiment_dir,
                       subset='train_small')

    '''run experiments'''
    experiments.run(tracker, visualize=False)
    experiments.report([tracker.name], return_report=False)
