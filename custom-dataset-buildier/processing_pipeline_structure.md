```json
{
  "face-filter-data": {
    "0-real": {
      "test": {
        "video_id_1": {
          "label": "0-real",
          "frames": [
            "datasets/face-filter-data/original/frames/video_id_1/frame_id_1.png",
            "datasets/face-filter-data/original/frames/video_id_1/frame_id_2.png"
            // a total of 32 frames
          ] 
        },
        "video_id_2": {
          "label": "0-real",
          "frames": [
            "datasets/face-filter-data/original/frames/video_id_2/frame_id_1.png",
            "datasets/face-filter-data/original/frames/video_id_2/frame_id_2.png"
            // a total of 32 frames
          ] 
        }
      }
    },
    "1-fake": {
      "test": {
        "video_id_1": {
          "label": "1-fake",
          "frames": [
            "datasets/face-filter-data/filtered/frames/video_id_1/frame_id_1.png",
            "datasets/face-filter-data/filtered/frames/video_id_1/frame_id_2.png"
            // a total of 32 frames
          ] 
        },
        "video_id_2": {
          "label": "1-fake",
          "frames": [
            "datasets/face-filter-data/filtered/frames/video_id_2/frame_id_1.png",
            "datasets/face-filter-data/filtered/frames/video_id_2/frame_id_2.png"
            // a total of 32 frames
          ] 
        }
      }
    }
  }
}
```

- Explanation:
- The dataset is organized into two main categories: "0-real" and "1-fake", representing real and manipulated videos, respectively.
- The original video frames belonging to the same video are grouped together under unique video IDs.
- Each video ID contains a list of frames, with each frame represented by its file path.
- 