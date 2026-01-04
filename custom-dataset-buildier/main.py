#!/usr/bin/env python3
"""
Face Filter Dataset Builder

This script creates a test dataset for deepfake detection models
by processing videos and labeling them as real (unfiltered) or fake (filtered).

Usage:
    # Add a real (unfiltered) video:
    python main.py --video path/to/video.mp4 --label real --dataset-name my-filter-test
    
    # Add a fake (filtered) video:
    python main.py --video path/to/video.mp4 --label fake --dataset-name my-filter-test
    
    # Batch processing from a file:
    python main.py --batch-file videos.txt --dataset-name my-filter-test
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np


def center_crop_and_resize(frame: np.ndarray, target_size: int = 256) -> np.ndarray:
    """
    Center crop the frame to a square and resize to target_size x target_size.
    Preserves aspect ratio by cropping the center portion.
    
    Args:
        frame: Input frame (H, W, C)
        target_size: Output size (default 256x256)
    
    Returns:
        Processed frame of shape (target_size, target_size, C)
    """
    h, w = frame.shape[:2]
    
    # Determine the size of the square crop (minimum of height/width)
    crop_size = min(h, w)
    
    # Calculate crop coordinates (center crop)
    start_y = (h - crop_size) // 2
    start_x = (w - crop_size) // 2
    
    # Crop the center square
    cropped = frame[start_y:start_y + crop_size, start_x:start_x + crop_size]
    
    # Resize to target size
    resized = cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_AREA)
    
    return resized


def extract_frames(
    video_path: str,
    output_dir: str,
    num_frames: int = 32,
    target_size: int = 256
) -> list[str]:
    """
    Extract evenly distributed frames from a video.
    
    Args:
        video_path: Path to the input video
        output_dir: Directory to save extracted frames
        num_frames: Number of frames to extract (default 32)
        target_size: Size of output frames (default 256)
    
    Returns:
        List of saved frame paths (relative to workspace root)
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames < num_frames:
        print(f"Warning: Video has only {total_frames} frames, extracting all of them")
        frame_indices = list(range(total_frames))
    else:
        # Calculate evenly distributed frame indices
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    saved_paths = []
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if not ret:
            print(f"Warning: Could not read frame {idx} from {video_path}")
            continue
        
        # Process frame: center crop and resize
        processed_frame = center_crop_and_resize(frame, target_size)
        
        # Save frame with zero-padded index
        frame_filename = f"{idx:03d}.png"
        frame_path = os.path.join(output_dir, frame_filename)
        cv2.imwrite(frame_path, processed_frame)
        
        # Store relative path (from workspace root)
        saved_paths.append(frame_path)
    
    cap.release()
    
    return saved_paths


def process_video(
    video_path: str,
    label: str,
    dataset_name: str,
    video_id: str,
    output_base: str = "datasets",
    num_frames: int = 32,
    target_size: int = 256
) -> dict:
    """
    Process a single video with the given label.
    
    Args:
        video_path: Path to the video
        label: Either "real" or "fake"
        dataset_name: Name of the dataset
        video_id: Unique identifier for this video
        output_base: Base directory for output (default "datasets")
        num_frames: Number of frames to extract per video
        target_size: Size of output frames
    
    Returns:
        Dictionary containing metadata for the video
    """
    # Define output directory based on label
    if label == "real":
        subfolder = "original"
    else:
        subfolder = "filtered"
    
    output_dir = os.path.join(output_base, dataset_name, subfolder, "frames", video_id)
    
    print(f"Processing video: {video_path}")
    print(f"  Label: {label}")
    print(f"  Video ID: {video_id}")
    
    # Extract frames
    frames = extract_frames(
        video_path, 
        output_dir, 
        num_frames=num_frames,
        target_size=target_size
    )
    print(f"  Extracted {len(frames)} frames")
    
    return {
        "video_id": video_id,
        "label": label,
        "frames": frames
    }


def load_existing_json(json_path: str, dataset_name: str) -> dict:
    """
    Load existing dataset JSON or create a new structure.
    
    Args:
        json_path: Path to the JSON file
        dataset_name: Name of the dataset
    
    Returns:
        Dataset JSON structure
    """
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            return json.load(f)
    
    # Create new structure
    return {
        dataset_name: {
            "0-real": {
                "test": {}
            },
            "1-fake": {
                "test": {}
            }
        }
    }


def add_video_to_json(
    dataset_json: dict,
    video_data: dict,
    dataset_name: str
) -> dict:
    """
    Add a processed video to the dataset JSON structure.
    
    Args:
        dataset_json: Existing dataset JSON structure
        video_data: Processed video data
        dataset_name: Name of the dataset
    
    Returns:
        Updated dataset JSON structure
    """
    label = video_data["label"]
    video_id = video_data["video_id"]
    
    # Determine which category to add to
    if label == "real":
        category = "0-real"
        label_str = "0-real"
    else:
        category = "1-fake"
        label_str = "1-fake"
    
    # Get existing videos in this category to determine the video key
    existing_videos = dataset_json[dataset_name][category]["test"]
    video_key = f"video_{len(existing_videos):04d}"
    
    # Add the video entry
    dataset_json[dataset_name][category]["test"][video_key] = {
        "label": label_str,
        "frames": video_data["frames"]
    }
    
    return dataset_json


def parse_batch_file(batch_file: str) -> list[tuple[str, str]]:
    """
    Parse a batch file containing videos and their labels.
    
    Expected format (one video per line):
        video_path.mp4,real
        video_path.mp4,fake
    
    Args:
        batch_file: Path to the batch file
    
    Returns:
        List of (video_path, label) tuples
    """
    videos = []
    
    with open(batch_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Try comma first, then tab
            if ',' in line:
                parts = [p.strip() for p in line.split(',')]
            elif '\t' in line:
                parts = [p.strip() for p in line.split('\t')]
            else:
                print(f"Warning: Skipping line {line_num}, invalid format: {line}")
                continue
            
            if len(parts) != 2:
                print(f"Warning: Skipping line {line_num}, expected 'video_path,label', got: {line}")
                continue
            
            video_path, label = parts
            label = label.lower()
            
            if label not in ['real', 'fake']:
                print(f"Warning: Skipping line {line_num}, label must be 'real' or 'fake', got: {label}")
                continue
            
            if not os.path.exists(video_path):
                print(f"Warning: Video not found: {video_path}")
                continue
            
            videos.append((video_path, label))
    
    return videos


def main():
    parser = argparse.ArgumentParser(
        description="Build a face filter test dataset for deepfake detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Add a single real (unfiltered) video:
    python main.py --video original.mp4 --label real --dataset-name face-filter-test

    # Add a single fake (filtered) video:
    python main.py --video filtered.mp4 --label fake --dataset-name face-filter-test
    
    # Process multiple videos from a batch file:
    python main.py --batch-file videos.txt --dataset-name face-filter-test
    
    # Batch file format (videos.txt):
    # video_path.mp4,real
    # another_video.mp4,fake
    # filtered_video.mp4,fake
        """
    )
    
    # Input options (mutually exclusive groups)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--batch-file",
        type=str,
        help="Path to a file containing videos and labels (one per line: video_path,label)"
    )
    input_group.add_argument(
        "--video",
        type=str,
        help="Path to a single video file"
    )
    
    parser.add_argument(
        "--label",
        type=str,
        choices=['real', 'fake'],
        help="Label for the video: 'real' (unfiltered) or 'fake' (filtered). Required if --video is used."
    )
    
    parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        help="Name for the dataset (will be used in output paths and JSON)"
    )
    
    parser.add_argument(
        "--output-base",
        type=str,
        default="datasets",
        help="Base directory for output (default: datasets)"
    )
    
    parser.add_argument(
        "--json-output",
        type=str,
        default="dataset_jsons",
        help="Directory to save the JSON metadata file (default: dataset_jsons)"
    )
    
    parser.add_argument(
        "--num-frames",
        type=int,
        default=32,
        help="Number of frames to extract per video (default: 32)"
    )
    
    parser.add_argument(
        "--frame-size",
        type=int,
        default=256,
        help="Size of output frames in pixels (default: 256)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.video and not args.label:
        parser.error("--label is required when --video is specified")
    
    # Collect videos to process
    videos = []
    
    if args.batch_file:
        videos = parse_batch_file(args.batch_file)
        if not videos:
            print("Error: No valid videos found in batch file")
            sys.exit(1)
    else:
        if not os.path.exists(args.video):
            print(f"Error: Video not found: {args.video}")
            sys.exit(1)
        videos = [(args.video, args.label)]
    
    # Count videos by label
    real_count = sum(1 for _, label in videos if label == 'real')
    fake_count = sum(1 for _, label in videos if label == 'fake')
    
    print(f"\n{'='*60}")
    print(f"Face Filter Dataset Builder")
    print(f"{'='*60}")
    print(f"Dataset name: {args.dataset_name}")
    print(f"Output base: {args.output_base}")
    print(f"Total videos: {len(videos)}")
    print(f"  - Real (unfiltered): {real_count}")
    print(f"  - Fake (filtered): {fake_count}")
    print(f"Frames per video: {args.num_frames}")
    print(f"Frame size: {args.frame_size}x{args.frame_size}")
    print(f"{'='*60}\n")
    
    # Load existing JSON or create new
    os.makedirs(args.json_output, exist_ok=True)
    json_path = os.path.join(args.json_output, f"{args.dataset_name}.json")
    dataset_json = load_existing_json(json_path, args.dataset_name)
    
    # Process all videos
    processed_count = 0
    
    for video_path, label in videos:
        video_id = Path(video_path).stem  # Use video filename as ID
        
        try:
            video_data = process_video(
                video_path=video_path,
                label=label,
                dataset_name=args.dataset_name,
                video_id=video_id,
                output_base=args.output_base,
                num_frames=args.num_frames,
                target_size=args.frame_size
            )
            
            # Add to JSON structure
            dataset_json = add_video_to_json(dataset_json, video_data, args.dataset_name)
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            continue
    
    if processed_count == 0:
        print("Error: No videos were successfully processed")
        sys.exit(1)
    
    # Save updated JSON
    with open(json_path, 'w') as f:
        json.dump(dataset_json, f, indent=2)
    
    # Count total videos in dataset
    total_real = len(dataset_json[args.dataset_name]["0-real"]["test"])
    total_fake = len(dataset_json[args.dataset_name]["1-fake"]["test"])
    
    print(f"\n{'='*60}")
    print(f"Dataset update complete!")
    print(f"{'='*60}")
    print(f"Processed {processed_count} video(s) in this run")
    print(f"\nTotal videos in dataset '{args.dataset_name}':")
    print(f"  - Real (unfiltered): {total_real}")
    print(f"  - Fake (filtered): {total_fake}")
    print(f"\nFrames saved to: {args.output_base}/{args.dataset_name}/")
    print(f"JSON metadata saved to: {json_path}")
    print(f"\nTo use this dataset, add '{args.dataset_name}' to test_dataset in config/test.yaml")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
