from __future__ import absolute_import
from SiamRPNEval import TrackerSiamRPNEval
from train.experimentrgbt import ExperimentRGBT


if __name__ == '__main__':


    # If modality==1: use only RGB. If modality==2: RGB+IR
    modality = 1

    if modality == 1:
        net_path = 'train/experiments/SiamRPN_RGB/model/model_e4.pth'
        experiment_name = 'RGB'
    else:
        net_path = 'train/experiments/SiamRPN_RGBIR/model/model_e1.pth'
        experiment_name = 'RGBIR'


    tracker = TrackerSiamRPNEval(modality=modality,
                                 model_path=net_path)

    experiments = ExperimentRGBT('/home/zuern/datasets/thermal_tracking/RGB-T234/',
                       experiment_name=experiment_name,
                       subset='val')

    '''run experiments'''
    experiments.run(tracker, visualize=False)
    experiments.report([tracker.name], return_report=False)
