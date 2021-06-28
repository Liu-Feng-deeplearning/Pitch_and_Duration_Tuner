#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import numpy as np
import parselmouth
import pyreaper
import pyworld as pw


def plot_line(path, x_list, y_list, label_list):
  assert len(x_list) == len(y_list) == len(label_list)
  plt.title('Result Analysis')
  for x_data, y_data, label in zip(x_list, y_list, label_list):
    plt.plot(x_data, y_data, label=label)
    # plt.plot(x2, y2, color='red', label='predict')
  plt.legend()  # 显示图例
  plt.xlabel('frame-index')
  plt.ylabel('value')
  plt.savefig(path, format='png')
  plt.close()
  return


class F0Extractor:
  def __init__(self, method_type, sample_rate=24000, hop_size=240, min_f0=20,
               max_f0=600):
    self._max_f0 = max_f0
    self._min_f0 = min_f0
    self._sr = sample_rate
    self._hop_size = hop_size
    self._method_type = method_type
    print("init f0 extractor with {}/{}".format(min_f0, max_f0))
    return

  def basic_analysis(self, signal):
    assert -1 <= np.min(signal) <= np.max(signal) <= 1.0
    if self._method_type == "praat":
      sound = parselmouth.Sound(signal.astype(np.float64), self._sr, 0.0)
      time_step = 0.0025
      pitch = sound.to_pitch(
        time_step=time_step, pitch_floor=self._min_f0,
        pitch_ceiling=self._max_f0)
      f0 = pitch.selected_array['frequency']
      time = pitch.xs()
      unvoiced_value = 0
    elif self._method_type == "sptk":
      signal = signal * 32767
      signal = signal.astype(np.int16)
      pm_times, pm, f0_times, f0, corr = pyreaper.reaper(
        signal, self._sr, frame_period=0.0025, maxf0=self._max_f0,
        minf0=self._min_f0, unvoiced_cost=1.1)
      unvoiced_value = -1
      time = f0_times
    elif self._method_type == "world":
      frame_period = 10
      f0, t = pw.dio(
        signal.astype(np.float64), self._sr, frame_period=frame_period)
      unvoiced_value = 0
      time = t
    else:
      raise Exception("unvalid method type")
    return f0.reshape((-1)), time, unvoiced_value

  @staticmethod
  def _extract_vuv(signal, unvoiced_value):
    is_unvoiced = np.isclose(signal,
                             unvoiced_value * np.ones_like(signal),
                             atol=1e-2)
    is_voiced = np.logical_not(is_unvoiced)
    return is_voiced

  @staticmethod
  def _interpolate(signal, is_voiced):
    """Linearly interpolates the signal in unvoiced regions such that there are no discontinuities.

      Args:
          signal (np.ndarray[n_frames, feat_dim]): Temporal signal.
          is_voiced (np.ndarray[n_frames]<bool>): Boolean array indicating if each frame is voiced.

      Returns:
          (np.ndarray[n_frames, feat_dim]): Interpolated signal, same shape as signal.
      """
    n_frames = signal.shape[0]
    feat_dim = signal.shape[1]

    # Initialize whether we are starting the search in voice/unvoiced.
    in_voiced_region = is_voiced[0]

    last_voiced_frame_i = None
    for i in range(n_frames):
      if is_voiced[i]:
        if not in_voiced_region:
          # Current frame is voiced, but last frame was unvoiced.
          # This is the first voiced frame after an unvoiced sequence,
          # interpolate the unvoiced region.

          # If the signal starts with an unvoiced region then `last_voiced_frame_i` will be None.
          # Bypass interpolation and just set this first unvoiced region to the current voiced frame value.
          if last_voiced_frame_i is None:
            signal[:i + 1] = signal[i]

          # Use `np.linspace` to create a interpolate a region that
          # includes the bordering voiced frames.
          else:
            start_voiced_value = signal[last_voiced_frame_i]
            end_voiced_value = signal[i]

            unvoiced_region_length = (i + 1) - last_voiced_frame_i
            interpolated_region = np.linspace(start_voiced_value,
                                              end_voiced_value,
                                              unvoiced_region_length)
            interpolated_region = interpolated_region.reshape(
              (unvoiced_region_length, feat_dim))

            signal[last_voiced_frame_i:i + 1] = interpolated_region

        # Move pointers forward, we are waiting to find another unvoiced section.
        last_voiced_frame_i = i

      in_voiced_region = is_voiced[i]

    # If the signal ends with an unvoiced region then it would not have been caught in the loop.
    # Similar to the case with an unvoiced region at the start we can bypass the interpolation.
    if not in_voiced_region:
      signal[last_voiced_frame_i:] = signal[last_voiced_frame_i]
    return signal

  def extract_f0_by_frame(self, signal, interpolate=False):
    f0, x0, unvoiced_value = self.basic_analysis(signal)
    f0 = f0.reshape((-1, 1))
    if interpolate:
      vuv = self._extract_vuv(f0, unvoiced_value=unvoiced_value)
      f0 = self._interpolate(f0, vuv).reshape((-1))

    f0 = f0.reshape((-1))
    frame_num = len(signal) // self._hop_size + 1
    frame_time = np.arange(frame_num) * self._hop_size / self._sr
    f0_by_frame = np.interp(frame_time, x0, f0)
    return f0_by_frame

  def plot_two_f0(self, saved_path, signal1, label1, signal2, label2):
    x_list = []
    y_list = []
    label_list = []

    f0_frame = self.extract_f0_by_frame(signal1)
    x_list.append(np.arange(np.size(f0_frame)))
    y_list.append(f0_frame)
    label_list.append(label1)

    f0_frame = self.extract_f0_by_frame(signal2)
    x_list.append(np.arange(np.size(f0_frame)))
    y_list.append(f0_frame)
    label_list.append(label2)

    plot_line(saved_path, x_list, y_list, label_list)
    return


def __test_for_pitch():
  import librosa
  wav_path = "demo.wav"
  sig, sr = librosa.load(wav_path, sr=24000)

  print("test for praat pitch ...")
  _ext = F0Extractor(sample_rate=sr, hop_size=int(sr * 0.01),
                     method_type="praat", min_f0=60, max_f0=400)
  f0_frame = _ext.extract_f0_by_frame(sig)
  print("res shape:", np.shape(f0_frame))
  return


if __name__ == '__main__':
  __test_for_pitch()
