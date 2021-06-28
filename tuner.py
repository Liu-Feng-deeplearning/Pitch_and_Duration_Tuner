#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:liufeng
# datetime:2020/7/16 下午4:56

import os

import parselmouth
from parselmouth.praat import call


class AudioChanger:
  def __init__(self):
    """ wrapper for praat (parselmoutch)

    Examples:
    https://parselmouth.readthedocs.io/en/latest/examples/pitch_manipulation.html#

    Details:
    https://www.fon.hum.uva.nl/praat/manual/Types_of_objects.html
    """
    self._time_step = 0.01
    self._min_f0 = 45
    self._max_f0 = 600
    return

  def _init_manipulate(self, sound):
    manipulation = call(sound, "To Manipulation", 0.001, self._min_f0,
                        self._max_f0)
    return manipulation

  def change_pitch_by_factor(self, wav_path, tune_path, factor=1.0, start=None,
                             end=None):
    """ change fragment pitch by factor.

    Args:
      wav_path: init wav path.
      tune_path: changed wav path.
      factor: coef, change pitch to pitch*factor
      start: start point, default is sound.xmin
      end: end point, default is sound.xmax

    Returns:

    """
    sound = parselmouth.Sound(wav_path)
    manipulation = self._init_manipulate(sound)
    start = start if start else sound.xmin
    end = end if end else sound.xmax
    pitch_tier = call(manipulation, "Extract pitch tier")
    call(pitch_tier, "Multiply frequencies", start, end, factor)
    call([pitch_tier, manipulation], "Replace pitch tier")
    self._save_sound(tune_path, manipulation)
    return

  def change_pitch_by_point(self, wav_path, tune_path, point_pitch):
    """ change audio pitch point by point.

    Args:
      wav_path: init wav path.
      tune_path: changed wav path.
      point_pitch: changed wav pitch. [(time_1, pitch_1), (time_2, pitch_2)...]

    Returns:

    """
    sound = parselmouth.Sound(wav_path)
    manipulation = self._init_manipulate(sound)

    pitch_tier = call("Create PitchTier...", "pitch_tier", 0.0, sound.xmax)
    for time, pitch in point_pitch:
      call(pitch_tier, "Add point...", time, pitch)
    call([pitch_tier, manipulation], "Replace pitch tier")
    self._save_sound(tune_path, manipulation)
    return

  def change_pitch_by_shift(self, wav_path, tune_path, shift=0, start=None,
                            end=None):
    """ change fragment pitch by shift.

    Args:
      wav_path: init wav path.
      tune_path: changed wav path.
      shift: coef, change pitch to pitch+shift
      start: start point, default is sound.xmin
      end: end point, default is sound.xmax

    Returns:

    """
    sound = parselmouth.Sound(wav_path)
    manipulation = self._init_manipulate(sound)
    start = start if start else sound.xmin
    end = end if end else sound.xmax
    pitch_tier = call(manipulation, "Extract pitch tier")
    call(pitch_tier, "Shift frequencies", start, end, shift, "Hertz")
    call([pitch_tier, manipulation], "Replace pitch tier")
    self._save_sound(tune_path, manipulation)
    return

  def change_dur_by_point(self, wav_path, tune_path, point_factor):
    """ change duration by point

    Args:
      wav_path: init wav path.
      tune_path: changed wav path.
      point_factor: [(points, factor)] change duration by factor at points

    Note:
      duration is changed continually. For example, if the total duration of
      some audio is 1s, and set point_factor [(0.0, 1.0), (0.3, 2.0), (1.0, 1.0)],
      the tuned audio will has duration of 1.5s(=(1.0+2.0)*0.3*0.5+(1.0+2.0)*0.7*0.5)

    Returns:

    """
    sound = parselmouth.Sound(wav_path)
    manipulation = self._init_manipulate(sound)
    dur_tier = call(manipulation, "Extract duration tier")
    for time, factor in point_factor:
      call(dur_tier, "Add point...", time, factor)
    call([dur_tier, manipulation], "Replace duration tier")
    self._save_sound(tune_path, manipulation)
    return

  @staticmethod
  def _save_sound(new_path, manipulation):
    new_sound = call(manipulation, "Get resynthesis (overlap-add)")
    new_sound.save(new_path, "WAV")
    return


def test():
  """test for tuner"""
  import shutil
  from pitch import F0Extractor
  import librosa

  demo_dir = "/workspace/project-nas-10487-sh/liufeng/huya_fast_vc/tools/demo"
  wav_path = "/workspace/cpfs-data/liufeng/vc_feature_v0/bzn/sp_wav/" \
             "sp-1.0-000001.wav"
  shutil.copy(wav_path, os.path.join(demo_dir, "init.wav"))
  audio_change = AudioChanger()
  f0_ext = F0Extractor(method_type="praat", sample_rate=24000, hop_size=240,
                       min_f0=45, max_f0=600)

  print("test for change_dur_by_point:")
  demo_path = os.path.join(demo_dir, "test_dur.wav")
  audio_change.change_dur_by_point(wav_path, demo_path,
                                   [(0.5, 1.0), (1.0, 2.0), (1.5, 1.0)])

  print("test for change_pitch_by_factor:")
  demo_path = os.path.join(demo_dir, "test_pitch_by_factor.wav")
  audio_change.change_pitch_by_factor(wav_path, demo_path,
                                      factor=1.1, start=0.5, end=1.5)
  sig1, _ = librosa.load(wav_path, sr=24000)
  sig2, _ = librosa.load(demo_path, sr=24000)
  f0_ext.plot_two_f0(os.path.join(demo_dir, "res.png"), sig1, "x1", sig2, "x2")

  print("test for change_pitch_by_point:")
  demo_path = os.path.join(demo_dir, "test_pitch_by_point.wav")
  sig1, _ = librosa.load(wav_path, sr=24000)
  pitch, time, _ = f0_ext.basic_analysis(sig1)
  point_pitch = [(x, y) for x, y in zip(time, pitch)]
  for idx in range(len(point_pitch)):
    if 0.5 < point_pitch[idx][0] < 1.5 and point_pitch[idx][1] > 1.0:
      point_pitch[idx] = (point_pitch[idx][0], point_pitch[idx][1] * 1.2)
  audio_change.change_pitch_by_point(wav_path, demo_path, point_pitch)
  sig2, _ = librosa.load(demo_path, sr=24000)
  f0_ext.plot_two_f0(os.path.join(demo_dir, "test_pitch_by_point.png"),
                     sig1, "x1", sig2, "x2")
  return


if __name__ == '__main__':
  test()
