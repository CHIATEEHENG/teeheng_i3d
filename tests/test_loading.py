import copy
import os
import os.path as osp

import numpy as np

from mmaction.datasets.pipelines import (DecordDecode, FrameSelector,
                                         OpenCVDecode, PyAVDecode,
                                         SampleFrames)


class TestLoading(object):

    @staticmethod
    def check_keys_contain(result_keys, target_keys):
        """Check if all elements in target_keys is in result_keys."""
        return set(target_keys).issubset(set(result_keys))

    @classmethod
    def setup_class(cls):
        cls.img_path = osp.join(osp.dirname(__file__), 'data/test.jpg')
        cls.video_path = osp.join(osp.dirname(__file__), 'data/test.mp4')
        cls.img_dir = osp.join(osp.dirname(__file__), 'data/test_imgs')
        cls.total_frames = len(os.listdir(cls.img_dir))
        cls.filename_tmpl = 'img_{:05}.jpg'
        cls.video_results = dict(filename=cls.video_path, label=1)
        cls.frame_results = dict(
            frame_dir=cls.img_dir,
            total_frames=cls.total_frames,
            filename_tmpl=cls.filename_tmpl,
            label=1)

    def test_sample_frames(self):
        target_keys = [
            'frame_inds', 'clip_len', 'frame_interval', 'num_clips',
            'total_frames'
        ]

        video_result = copy.deepcopy(self.video_results)
        frame_result = copy.deepcopy(self.frame_results)
        config = dict(
            clip_len=3, frame_interval=1, num_clips=1, temporal_jitter=False)
        sample_frames = SampleFrames(**config)
        sample_frames_results = sample_frames(video_result)
        assert self.check_keys_contain(sample_frames_results.keys(),
                                       target_keys)
        assert len(sample_frames_results['frame_inds']) == 3
        sample_frames_results = sample_frames(frame_result)
        assert len(sample_frames_results['frame_inds']) == 3

        video_result = copy.deepcopy(self.video_results)
        frame_result = copy.deepcopy(self.frame_results)
        config = dict(
            clip_len=3, frame_interval=1, num_clips=1, temporal_jitter=True)
        sample_frames = SampleFrames(**config)
        sample_frames_results = sample_frames(video_result)
        assert self.check_keys_contain(sample_frames_results.keys(),
                                       target_keys)
        assert len(sample_frames_results['frame_inds']) == 3
        sample_frames_results = sample_frames(frame_result)
        assert len(sample_frames_results['frame_inds']) == 3

    def test_pyav_decode(self):
        target_keys = ['frame_inds', 'imgs', 'ori_shape']

        video_result = copy.deepcopy(self.video_results)
        video_result['frame_inds'] = np.arange(1, self.total_frames, 5)
        pyav_decode = PyAVDecode()
        pyav_decode_result = pyav_decode(video_result)
        assert self.check_keys_contain(pyav_decode_result.keys(), target_keys)
        assert pyav_decode_result['ori_shape'] == (256, 340)

        video_result = copy.deepcopy(self.video_results)
        video_result['frame_inds'] = np.arange(1, self.total_frames, 5)
        pyav_decode = PyAVDecode(multi_thread=True)
        pyav_decode_result = pyav_decode(video_result)
        assert self.check_keys_contain(pyav_decode_result.keys(), target_keys)
        assert pyav_decode_result['ori_shape'] == (256, 340)

    def test_decord_decode(self):
        target_keys = ['frame_inds', 'imgs', 'ori_shape']

        video_result = copy.deepcopy(self.video_results)
        video_result['frame_inds'] = np.arange(1, self.total_frames, 5)
        decord_decode = DecordDecode()
        decord_decode_result = decord_decode(video_result)
        assert self.check_keys_contain(decord_decode_result.keys(),
                                       target_keys)
        assert decord_decode_result['ori_shape'] == (256, 340)

    def test_opencv_decode(self):
        target_keys = ['frame_inds', 'imgs', 'ori_shape']

        video_result = copy.deepcopy(self.video_results)
        video_result['frame_inds'] = np.arange(1, self.total_frames, 5)
        opencv_decode = OpenCVDecode()
        opencv_decode_result = opencv_decode(video_result)
        assert self.check_keys_contain(opencv_decode_result.keys(),
                                       target_keys)
        assert opencv_decode_result['ori_shape'] == (256, 340)

    def test_frame_selector(self):
        target_keys = ['frame_inds', 'imgs', 'ori_shape']

        frame_result = copy.deepcopy(self.frame_results)
        frame_result['frame_inds'] = np.arange(1, self.total_frames, 5)
        frame_selector = FrameSelector()
        frame_selector_result = frame_selector(frame_result)
        assert self.check_keys_contain(frame_selector_result.keys(),
                                       target_keys)
        assert frame_selector_result['ori_shape'] == (240, 320)
