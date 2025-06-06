from tqdm import tqdm
import cv2
import os

def break_down_into_folder(video, output_dir, step, clip_size, clip_step, start_idx, max_idx):
    """
    Splits `video` into multiple clips. Each clip starts every `step` frames,
    contains `clip_size` frames sampled every `clip_step` frames, and only frames
    between `start_idx` and `max_idx` (inclusive) are considered.
    All output images are saved together under <output_dir>/<video_basename>/images/.
    """
    # Prepare paths
    base_name = os.path.splitext(os.path.basename(video))[0]
    root_dir = os.path.join(output_dir, base_name)
    images_dir = os.path.join(root_dir)
    os.makedirs(images_dir, exist_ok=True)

    # Open video
    vid = cv2.VideoCapture(video)
    if not vid.isOpened():
        raise IOError(f"Cannot open video file {video}")

    # Count total frames
    total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_idx < 0 or max_idx >= total_frames:
        max_idx = total_frames - 1

    clip_count = 0
    # Iterate over each clip start frame
    for clip_start in tqdm(range(start_idx, max_idx + 1, step)):
        for i in range(clip_size):
            frame_idx = clip_start + i * clip_step
            if frame_idx > max_idx:
                break

            vid.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, frame = vid.read()
            if not success:
                # print(f"[Warning] failed to read frame {frame_idx}, stopping clip #{clip_count}")
                break

            out_path = os.path.join(images_dir, f"frame{frame_idx:06d}.jpg")
            cv2.imwrite(out_path, frame)
            # print(f"Saved frame {frame_idx} (from clip {clip_count})")

        clip_count += 1

    vid.release()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Break a video into multiple clips of sampled frames."
    )
    parser.add_argument(
        "--video", type=str, required=True,
        help="Path to the input video file."
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to save extracted images."
    )
    parser.add_argument(
        "--step", type=int, default=1000,
        help="Number of frames between the start of each clip."
    )
    parser.add_argument(
        "--clip_size", type=int, default=10,
        help="How many frames to extract per clip."
    )
    parser.add_argument(
        "--clip_step", type=int, default=7,
        help="Frame interval within each clip."
    )
    parser.add_argument(
        "--start_idx", type=int, default=0,
        help="Index of the first frame to consider."
    )
    parser.add_argument(
        "--max_idx", type=int, default=-1,
        help="Index of the last frame to consider; -1 means the end of the video."
    )

    args = parser.parse_args()
    break_down_into_folder(
        video=args.video,
        output_dir=args.output_dir,
        step=args.step,
        clip_size=args.clip_size,
        clip_step=args.clip_step,
        start_idx=args.start_idx,
        max_idx=args.max_idx
    )
