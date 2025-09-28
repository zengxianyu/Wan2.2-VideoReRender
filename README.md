# Wan2.2-VideoReRender

Examples:

```
  # Use the default example
  python wan22_style.py

  # Process a single video file
  python wan22_style.py --input video.mp4 --output processed_video.mp4
  
  # Process all videos in a directory
  python wan22_style.py --input /path/to/videos/ --output /path/to/output/
  
  # Process with custom prompts
  python wan22_style.py --input video.mp4 --positive "A cinematic scene" --negative "blurry, low quality"
  
  # Process with style prompt
  python wan22_style.py --input video.mp4 --style-positive "in the style of Van Gogh" --positive "A beautiful landscape"
  
  # Process directory with custom output directory
  python wan22_style.py --input /path/to/videos/ --output /path/to/output/ --batch
```

Example results:

![dog](https://cdn.openart.ai/workflow_thumbnails/AfjGcZUOHfgKmNUiv1RK/gif_Uv2JAkyM_1759047750755_raw.gif)

<video controls>
  <source src="https://cdn.openart.ai/workflow_thumbnails/AfjGcZUOHfgKmNUiv1RK/video_cXmzdrta_1759039932523_raw.mp4" type="video/mp4">
</video>

<video controls>
  <source src="https://cdn.openart.ai/workflow_thumbnails/AfjGcZUOHfgKmNUiv1RK/video_ZnkSJDiZ_1759039932435_raw.mp4" type="video/mp4">
</video>

<video controls>
  <source src="https://cdn.openart.ai/workflow_thumbnails/AfjGcZUOHfgKmNUiv1RK/video_3lLDJdP-_1759098438021_raw.mp4" type="video/mp4">
</video>
