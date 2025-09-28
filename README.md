# Wan2.2-VideoReRender

## ComfyUI
[Workflow](https://github.com/zengxianyu/Wan2.2-VideoReRender/blob/main/ComfyUI/user/default/workflows/video-style-flux-wan2.2fun.json)

## Commandline

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

## example

![dog](https://github.com/zengxianyu/zengxianyu/blob/main/dogconcat_gradio_compatible.gif?raw=true)
![arm](https://github.com/zengxianyu/zengxianyu/blob/main/armconcat_gradio_compatible.gif?raw=true)
![west](https://github.com/zengxianyu/zengxianyu/blob/main/westconcat_gradio_compatible.gif?raw=true)
![town4](https://github.com/zengxianyu/zengxianyu/blob/main/town4concat_gradio_compatible.gif?raw=true)
