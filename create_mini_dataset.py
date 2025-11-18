import json

def create_mini_dataset(input_json, output_json, num_real=2, num_fake=2):
    """
    Create a mini version of the dataset with just a few videos for quick testing.

    Args:
        input_json: Path to full dataset JSON
        output_json: Path to save mini dataset JSON
        num_real: Number of real videos to include
        num_fake: Number of fake videos to include
    """
    # Load the full dataset
    with open(input_json, 'r') as f:
        full_data = json.load(f)

    dataset_name = list(full_data.keys())[0]

    # Extract the mini dataset name from output filename
    # e.g., "dataset_jsons/Celeb-DF-v1-mini.json" -> "Celeb-DF-v1-mini"
    import os
    mini_dataset_name = os.path.splitext(os.path.basename(output_json))[0]

    # Create mini dataset structure with the new name
    mini_data = {
        mini_dataset_name: {
            "0-real": {"test": {}},
            "1-fake": {"test": {}}
        }
    }

    # Get first N real videos
    real_videos = list(full_data[dataset_name]["0-real"]["test"].items())[:num_real]
    for video_id, video_info in real_videos:
        mini_data[mini_dataset_name]["0-real"]["test"][video_id] = video_info

    # Get first N fake videos
    fake_videos = list(full_data[dataset_name]["1-fake"]["test"].items())[:num_fake]
    for video_id, video_info in fake_videos:
        mini_data[mini_dataset_name]["1-fake"]["test"][video_id] = video_info

    # Save mini dataset
    with open(output_json, 'w') as f:
        json.dump(mini_data, f, indent=2)

    print(f"✓ Created mini dataset: {output_json}")
    print(f"  Real videos: {num_real}")
    print(f"  Fake videos: {num_fake}")

if __name__ == "__main__":
    create_mini_dataset(
        input_json="dataset_jsons/Celeb-DF-v1.json",
        output_json="dataset_jsons/Celeb-DF-v1-mini.json",
        num_real=10,  # Just 2 real videos
        num_fake=10   # Just 2 fake videos
    )