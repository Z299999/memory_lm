"""
Video generation module.

Only imported if make_video is True.
"""

from collections import deque
from pathlib import Path
from typing import List, Optional
import numpy as np


def save_video(
    frames: List[np.ndarray],
    output_path: Path,
    fps: int = 24,
):
    """
    Save frames as MP4 video.

    Args:
        frames: List of (H, W, 3) uint8 numpy arrays
        output_path: Path to output video file
        fps: Frames per second
    """
    # Import imageio here to keep it lazy
    import imageio

    if not frames:
        print("Warning: No frames to save")
        return

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Use imageio to write video
    # imageio-ffmpeg plugin handles MP4 encoding
    writer = imageio.get_writer(
        str(output_path),
        fps=fps,
        codec="libx264",
        quality=8,  # Quality setting (0-10, higher is better)
        pixelformat="yuv420p",  # For broad compatibility
    )

    for frame in frames:
        writer.append_data(frame)

    writer.close()
    print(f"Video saved to {output_path}")


def save_video_incremental(
    output_path: Path,
    fps: int = 24,
):
    """
    Create an incremental video writer for memory efficiency.

    Returns a writer object that can be used to add frames one at a time.

    Args:
        output_path: Path to output video file
        fps: Frames per second

    Returns:
        Video writer object with append_data and close methods
    """
    import imageio

    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = imageio.get_writer(
        str(output_path),
        fps=fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
    )

    return writer


class VideoWriter:
    """
    Wrapper for incremental video writing.

    Supports optional ring buffer mode to save only the last N seconds.
    """

    def __init__(
        self,
        output_path: Path,
        fps: int = 24,
        save_last_seconds: int = 0,
    ):
        """
        Initialize video writer.

        Args:
            output_path: Path to output video file
            fps: Frames per second
            save_last_seconds: If > 0, only save the last N seconds of frames.
                              Uses a ring buffer to limit memory usage.
                              If 0, saves all frames (writes incrementally).
        """
        self.output_path = output_path
        self.fps = fps
        self.save_last_seconds = save_last_seconds
        self._writer = None
        self._frame_count = 0

        # Ring buffer mode
        if save_last_seconds > 0:
            max_frames = fps * save_last_seconds
            self._frame_buffer: Optional[deque] = deque(maxlen=max_frames)
            self._use_buffer = True
        else:
            self._frame_buffer = None
            self._use_buffer = False

    def start(self):
        """Start the video writer (only for non-buffer mode)."""
        if self._use_buffer:
            # In buffer mode, we don't start the writer until close()
            return

        import imageio

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self._writer = imageio.get_writer(
            str(self.output_path),
            fps=self.fps,
            codec="libx264",
            quality=8,
            pixelformat="yuv420p",
        )

    def add_frame(self, frame: np.ndarray):
        """
        Add a frame to the video.

        Args:
            frame: (H, W, 3) uint8 numpy array
        """
        self._frame_count += 1

        if self._use_buffer:
            # Add to ring buffer (old frames automatically discarded)
            self._frame_buffer.append(frame.copy())
        else:
            # Write incrementally
            if self._writer is None:
                self.start()
            self._writer.append_data(frame)

    def close(self):
        """Close the video writer and finalize the file."""
        if self._use_buffer:
            # Write buffered frames to video
            if self._frame_buffer and len(self._frame_buffer) > 0:
                import imageio

                self.output_path.parent.mkdir(parents=True, exist_ok=True)

                writer = imageio.get_writer(
                    str(self.output_path),
                    fps=self.fps,
                    codec="libx264",
                    quality=8,
                    pixelformat="yuv420p",
                )

                for frame in self._frame_buffer:
                    writer.append_data(frame)

                writer.close()
                print(
                    f"Video saved to {self.output_path} "
                    f"({len(self._frame_buffer)} frames, last {self.save_last_seconds}s of {self._frame_count} total)"
                )
            else:
                print("Warning: No frames to save")

            self._frame_buffer = None
        else:
            if self._writer is not None:
                self._writer.close()
                self._writer = None
                print(f"Video saved to {self.output_path} ({self._frame_count} frames)")

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
