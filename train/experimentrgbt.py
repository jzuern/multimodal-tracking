from __future__ import absolute_import, division, print_function

import os
import numpy as np
import glob
import json
import time
import matplotlib
import matplotlib.pyplot as plt
import six

from PIL import Image
from got10k.utils.metrics import rect_iou
from got10k.utils.viz import show_frame



class RGBTSequence():

    def __init__(self, datadir, subset='test'):

        self.anno_files = []
        self.seq_dirs = []
        self.seq_names = []
        self.subset = subset


        sequence_list = sorted(glob.glob(datadir + '/*'))

        split_idx = 20
        # split_idx = 5

        if subset == 'test' or subset == 'val':
            sequence_list = sequence_list[0:split_idx]
        elif subset == 'train':
            sequence_list = sequence_list[split_idx:]
        elif subset == 'train_small':
            sequence_list = sequence_list[split_idx:split_idx+20]


        for sequence in sequence_list:
            self.anno_files.append(sequence + '/visible.txt')
            self.seq_dirs.append(sequence)
            self.seq_names.append(sequence.split('/')[-1])

    def __len__(self):
        return len(self.seq_names)

    def __getitem__(self, index):

        if isinstance(index, six.string_types):
            if not index in self.seq_names:
                raise Exception('Sequence {} not found.'.format(index))
            index = self.seq_names.index(index)


        img_files_rgb = sorted(glob.glob(os.path.join(
            self.seq_dirs[index], 'visible','*.jpg')))

        img_files_ir = sorted(glob.glob(os.path.join(
            self.seq_dirs[index], 'infrared', '*.jpg')))

        anno = np.loadtxt(self.anno_files[index], delimiter=',')


        if self.subset == 'test' and anno.ndim == 1:
            assert len(anno) == 4
            anno = anno[np.newaxis, :]
        else:
            assert len(img_files_rgb) == len(img_files_ir)
            assert len(img_files_rgb) == len(anno)

        return img_files_rgb, img_files_ir, anno





class ExperimentRGBT(object):
    r"""Experiment pipeline and evaluation toolkit for GOT-10k dataset.

    Args:
        root_dir (string): Root directory of GOT-10k dataset where
            ``train``, ``val`` and ``test`` folders exist.
        subset (string): Specify ``train``, ``val`` or ``test``
            subset of GOT-10k.
        list_file (string, optional): If provided, only run experiments on
            sequences specified by this file.
        result_dir (string, optional): Directory for storing tracking
            results. Default is ``./results``.
        report_dir (string, optional): Directory for storing performance
            evaluation results. Default is ``./reports``.
    """

    def __init__(self, root_dir, experiment_name, experiment_dir, subset='val', list_file=None,
                 result_dir='results', report_dir='reports'):

        super(ExperimentRGBT, self).__init__()

        self.subset = subset
        self.dataset = RGBTSequence( root_dir, subset=subset)

        self.result_dir = experiment_dir
        self.report_dir = experiment_dir

        self.nbins_iou = 101
        self.nbins_ce = 101

        self.repetitions = 3
        self.experiment_name = experiment_name


    def run(self, tracker, visualize=False):

        if self.subset == 'test':
            time.sleep(1)

        print('Running tracker %s on RGBT dataset...' % tracker.name)
        self.dataset.return_meta = False

        # loop over the complete dataset
        for s, (img_rgb_files, img_ir_files, anno) in enumerate(self.dataset):

            seq_name = self.dataset.seq_names[s]

            print('--Sequence %d/%d: %s' % (s + 1, len(self.dataset), seq_name))

            # run multiple repetitions for each sequence
            for r in range(self.repetitions):
                # check if the tracker is deterministic
                if r > 0 and tracker.is_deterministic:
                    break
                elif r == 3 and self._check_deterministic(
                        tracker.name, seq_name):
                    print('  Detected a deterministic tracker, ' +
                          'skipping remaining trials.')
                    break
                print(' Repetition: %d' % (r + 1))

                # skip if results exist
                record_file = os.path.join(
                    self.result_dir, self.experiment_name, '%s_%03d.txt' % (seq_name, r + 1))

                if os.path.exists(record_file):
                    print('  Found results, skipping', seq_name)
                    continue


                # tracking loop
                boxes, times = tracker.track(img_rgb_files, img_ir_files, anno[0, :], visualize=visualize)


                # record results
                self._record(record_file, boxes, times)



    def report(self, tracker_names, return_report=False):

        assert isinstance(tracker_names, (list, tuple))

        if self.subset == 'val' or True:

            # meta information is useful when evaluation
            self.dataset.return_meta = True

            # assume tracker_names[0] is your tracker
            report_dir = os.path.join(self.report_dir, self.experiment_name)

            if not os.path.exists(report_dir):
                os.makedirs(report_dir)

            report_file = os.path.join(report_dir, 'performance.json')

            # visible ratios of all sequences
            seq_names = self.dataset.seq_names

            covers = {s: 8 for s in seq_names}

            performance = {}


            for name in tracker_names:

                print('Evaluating', name)
                ious = {}
                times = {}

                performance.update({name: {
                    'overall': {},
                    'seq_wise': {}}})


                for s, (_, _, anno) in enumerate(self.dataset):

                    seq_name = self.dataset.seq_names[s]
                    record_files = glob.glob(os.path.join(
                        self.result_dir, self.experiment_name, '%s_[0-9]*.txt' % seq_name))

                    print('Evaluating {}'.format(seq_name))

                    if len(record_files) == 0:
                        print('\tResults for {} not found. Skipping.'.format(seq_name))
                        continue
                    else:
                        print('\tEvaluating results for {}.'.format(seq_name))

                    # read results of all repetitions
                    boxes = [np.loadtxt(f, delimiter=',') for f in record_files]

                    assert all([b.shape == anno.shape for b in boxes])

                    # calculate and stack all ious
                    bound = np.array([630, 460])
                    seq_ious = [rect_iou(b[1:], anno[1:], bound=bound) for b in boxes]


                    # only consider valid frames where targets are visible
                    seq_ious = [t[covers[seq_name] > 0] for t in seq_ious]
                    seq_ious = np.concatenate(seq_ious)
                    ious[seq_name] = seq_ious

                    # stack all tracking times
                    times[seq_name] = []
                    time_file = os.path.join(
                        self.result_dir, self.experiment_name, '%s_time.txt' % seq_name)

                    if os.path.exists(time_file):
                        seq_times = np.loadtxt(time_file, delimiter=',')
                        seq_times = seq_times[~np.isnan(seq_times)]
                        seq_times = seq_times[seq_times > 0]
                        if len(seq_times) > 0:
                            times[seq_name] = seq_times
                    else:
                        print('\tCould not find times file {}'.format(time_file))

                    # store sequence-wise performance
                    ao, sr, speed, _ = self._evaluate(seq_ious, seq_times)
                    performance[name]['seq_wise'].update({seq_name: {
                        'ao': ao,
                        'sr': sr,
                        'speed_fps': speed,
                        'length': len(anno) - 1}})

                # ious = np.concatenate(list(ious.values()))

                ious_list = [iou[0] for iou in ious.values()]
                ious = np.concatenate(ious_list)
                ious = np.expand_dims(ious, axis=0)

                times = np.concatenate(list(times.values()))

                # store overall performance
                ao, sr, speed, succ_curve = self._evaluate(ious, times)
                performance[name].update({'overall': {
                    'ao': ao,
                    'sr': sr,
                    'speed_fps': speed,
                    'succ_curve': succ_curve.tolist()}})

            # save performance
            with open(report_file, 'w') as f:
                json.dump(performance, f, indent=4)
            # plot success curves

            if return_report:
                return performance
            else:
                plotter = self.plot_curves([report_file], tracker_names)
                return


    def show(self, tracker_names, seq_names=None, play_speed=1):
        if seq_names is None:
            seq_names = self.dataset.seq_names
        elif isinstance(seq_names, str):
            seq_names = [seq_names]
        assert isinstance(tracker_names, (list, tuple))
        assert isinstance(seq_names, (list, tuple))

        play_speed = int(round(play_speed))
        assert play_speed > 0
        self.dataset.return_meta = False

        for s, seq_name in enumerate(seq_names):
            print('[%d/%d] Showing results on %s...' % (
                s + 1, len(seq_names), seq_name))

            # load all tracking results
            records = {}
            for name in tracker_names:
                record_file = os.path.join(
                    self.result_dir, name, seq_name,
                    '%s_001.txt' % seq_name)
                records[name] = np.loadtxt(record_file, delimiter=',')

            # loop over the sequence and display results
            img_files, anno = self.dataset[seq_name]
            for f, img_file in enumerate(img_files):
                if not f % play_speed == 0:
                    continue
                image = Image.open(img_file)
                boxes = [anno[f]] + [
                    records[name][f] for name in tracker_names]
                show_frame(image, boxes,
                           legends=['GroundTruth'] + tracker_names,
                           colors=['w', 'r', 'g', 'b', 'c', 'm', 'y',
                                   'orange', 'purple', 'brown', 'pink'])

    def _record(self, record_file, boxes, times):
        # record bounding boxes
        record_dir = os.path.dirname(record_file)
        if not os.path.isdir(record_dir):
            os.makedirs(record_dir)
        np.savetxt(record_file, boxes, fmt='%.3f', delimiter=',')
        print('  Results recorded at', record_file)

        # record running times
        time_file = record_file[:record_file.rfind('_')] + '_time.txt'
        times = times[:, np.newaxis]
        if os.path.exists(time_file):
            exist_times = np.loadtxt(time_file, delimiter=',')
            if exist_times.ndim == 1:
                exist_times = exist_times[:, np.newaxis]
            times = np.concatenate((exist_times, times), axis=1)
        np.savetxt(time_file, times, fmt='%.8f', delimiter=',')

    def _check_deterministic(self, tracker_name, seq_name):
        record_dir = os.path.join(
            self.result_dir, tracker_name, seq_name)
        record_files = sorted(glob.glob(os.path.join(
            record_dir, '%s_[0-9]*.txt' % seq_name)))

        if len(record_files) < 3:
            return False

        records = []
        for record_file in record_files:
            with open(record_file, 'r') as f:
                records.append(f.read())

        return len(set(records)) == 1

    def _evaluate(self, ious, times):

        # AO, SR and tracking speed
        ao = np.mean(ious)
        sr = np.mean(ious > 0.5)

        if len(times) > 0:
            # times has to be an array of positive values
            speed_fps = np.mean(1. / times)
        else:
            speed_fps = -1

        # success curve
        thr_iou = np.linspace(0, 1, 101)


        # bin_iou = np.greater(ious[:, None], thr_iou[None, :])
        bin_iou = np.greater(np.expand_dims(ious[0, :], axis=1), thr_iou[None, :])

        succ_curve = np.mean(bin_iou, axis=0)

        return ao, sr, speed_fps, succ_curve




    def plot_curves(self, report_files, tracker_names):

        assert isinstance(report_files, list), \
            'Expected "report_files" to be a list, ' \
            'but got %s instead' % type(report_files)

        # assume tracker_names[0] is your tracker
        report_dir = os.path.join(self.report_dir, self.experiment_name)
        if not os.path.exists(report_dir):
            os.makedirs(report_dir)

        performance = {}
        for report_file in report_files:
            with open(report_file) as f:
                performance.update(json.load(f))

        succ_file = os.path.join(report_dir, 'success_plot.png')
        key = 'overall'

        # sort trackers by AO
        tracker_names = list(performance.keys())
        aos = [t[key]['ao'] for t in performance.values()]
        inds = np.argsort(aos)[::-1]
        tracker_names = [tracker_names[i] for i in inds]

        # markers
        markers = ['-', '--', '-.']
        markers = [c + m for m in markers for c in [''] * 10]

        # # plot success curves
        thr_iou = np.linspace(0, 1, self.nbins_iou)
        fig, ax = plt.subplots()
        lines = []
        legends = []
        for i, name in enumerate(tracker_names):
            line, = ax.plot(thr_iou,
                            performance[name][key]['succ_curve'],
                            markers[i % len(markers)])
            lines.append(line)
            legends.append('%s: [%.3f]' % (
                name, performance[name][key]['ao']))
        matplotlib.rcParams.update({'font.size': 7.4})
        legend = ax.legend(lines, legends, loc='center left',
                           bbox_to_anchor=(1, 0.5))

        matplotlib.rcParams.update({'font.size': 9})
        ax.set(xlabel='Overlap threshold',
               ylabel='Success rate',
               xlim=(0, 1), ylim=(0, 1),
               title='Success plots on RGBT-234. Experiment: {}'.format(self.experiment_name))
        ax.grid(True)
        fig.tight_layout()

        print('Saving success plots to', succ_file)
        fig.savefig(succ_file,
                    bbox_extra_artists=(legend,),
                    bbox_inches='tight',
                    dpi=300)


        # # plot precision curves
        # thr_ce = np.arange(0, self.nbins_ce)
        # fig, ax = plt.subplots()
        # lines = []
        # legends = []
        # for i, name in enumerate(tracker_names):
        #     line, = ax.plot(thr_ce,
        #                     performance[name][key]['precision_curve'],
        #                     markers[i % len(markers)])
        #     lines.append(line)
        #     legends.append('%s: [%.3f]' % (name, performance[name][key]['precision_score']))
        # matplotlib.rcParams.update({'font.size': 7.4})
        # legend = ax.legend(lines, legends, loc='center left',
        #                    bbox_to_anchor=(1, 0.5))
        #
        # matplotlib.rcParams.update({'font.size': 9})
        # ax.set(xlabel='Location error threshold',
        #        ylabel='Precision',
        #        xlim=(0, thr_ce.max()), ylim=(0, 1),
        #        title='Precision plots of OPE')
        # ax.grid(True)
        # fig.tight_layout()
        #
        # print('Saving precision plots to', prec_file)
        # fig.savefig(prec_file, dpi=300)

        return None
        # return plt
