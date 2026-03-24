# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES enabling <CAT-K> or otherwise documented as
# NVIDIA-proprietary are not a contribution and subject to the following terms and conditions:
# SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import distutils.spawn
import distutils.version
import os
import os.path
import pkgutil
import subprocess

import numpy as np


class ImageEncoder(object):
    def __init__(self, output_path, frame_shape, frames_per_sec, output_frames_per_sec):
        self.proc = None
        self.output_path = output_path
        # Frame shape should be lines-first, so w and h are swapped
        h, w, pixfmt = frame_shape
        if pixfmt != 3 and pixfmt != 4:
            raise RuntimeError(
                "Your frame has shape {}, but we require (w,h,3) or (w,h,4), i.e., RGB values for a w-by-h image, with an optional alpha channel.".format(
                    frame_shape
                )
            )
        self.wh = (w, h)
        self.includes_alpha = pixfmt == 4
        self.frame_shape = frame_shape
        self.frames_per_sec = frames_per_sec
        self.output_frames_per_sec = output_frames_per_sec

        if distutils.spawn.find_executable("avconv") is not None:
            self.backend = "avconv"
        elif distutils.spawn.find_executable("ffmpeg") is not None:
            self.backend = "ffmpeg"
        elif pkgutil.find_loader("imageio_ffmpeg"):
            raise RuntimeError
            # import imageio_ffmpeg
            # self.backend = imageio_ffmpeg.get_ffmpeg_exe()
        else:
            raise RuntimeError(
                """Found neither the ffmpeg nor avconv executables. On OS X, you can install ffmpeg via `brew install ffmpeg`. On most Ubuntu variants, `sudo apt-get install ffmpeg` should do it. On Ubuntu 14.04, however, you'll need to install avconv with `sudo apt-get install libav-tools`. Alternatively, please install imageio-ffmpeg with `pip install imageio-ffmpeg`"""
            )

        self.start()

    @property
    def version_info(self):
        return {
            "backend": self.backend,
            "version": str(
                subprocess.check_output(
                    [self.backend, "-version"], stderr=subprocess.STDOUT
                )
            ),
            "cmdline": self.cmdline,
        }

    def start(self):
        self.cmdline = (
            self.backend,
            "-nostats",
            "-loglevel",
            "error",  # suppress warnings
            "-y",
            # input
            "-f",
            "rawvideo",
            "-s:v",
            "{}x{}".format(*self.wh),
            "-pix_fmt",
            ("rgb32" if self.includes_alpha else "rgb24"),
            "-framerate",
            "%d" % self.frames_per_sec,
            "-i",
            "-",  # this used to be /dev/stdin, which is not Windows-friendly
            # output
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-r",
            "%d" % self.output_frames_per_sec,
            self.output_path,
        )

        # print('Starting %s with "%s"', self.backend, " ".join(self.cmdline))
        if hasattr(os, "setsid"):  # setsid not present on Windows
            self.proc = subprocess.Popen(
                self.cmdline, stdin=subprocess.PIPE, preexec_fn=os.setsid
            )
        else:
            self.proc = subprocess.Popen(self.cmdline, stdin=subprocess.PIPE)

    def capture_frame(self, frame):
        if not isinstance(frame, (np.ndarray, np.generic)):
            raise RuntimeError(
                "Wrong type {} for {} (must be np.ndarray or np.generic)".format(
                    type(frame), frame
                )
            )
        if frame.shape != self.frame_shape:
            raise RuntimeError(
                "Your frame has shape {}, but the VideoRecorder is configured for shape {}.".format(
                    frame.shape, self.frame_shape
                )
            )
        if frame.dtype != np.uint8:
            raise RuntimeError(
                "Your frame has data type {}, but we require uint8 (i.e. RGB values from 0-255).".format(
                    frame.dtype
                )
            )

        try:
            if distutils.version.LooseVersion(
                np.__version__
            ) >= distutils.version.LooseVersion("1.9.0"):
                self.proc.stdin.write(frame.tobytes())
            else:
                self.proc.stdin.write(frame.tostring())
        except Exception as e:
            stdout, stderr = self.proc.communicate()
            print("VideoRecorder encoder failed: %s", stderr)

    def close(self):
        self.proc.stdin.close()
        ret = self.proc.wait()
        if ret != 0:
            print("VideoRecorder encoder exited with status {}".format(ret))
