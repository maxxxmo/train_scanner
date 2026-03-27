# DATA EXTRACTION
As I said I cant I extract everythind directly on py PC. I will also only use on_track images.
I have 2 functions:
- fast_streaming_conversion: Create output folders, open frames, filter 'on_track' frames only, Create parallel process_image_task
- process_image_task : Decode a single frame, letterboxing, clamping, transform data to yolo and save to jpg.

## process_image_task
- Decoding: Transform binary from tar file into a frame

- Letterboxing and clamping: 

- Yolo_conversion: Change coords to relative so resolution wont change anything.

- save frame in jpg, saves updated coords in a .txt file


## fast_streaming_conversion

- Open a single frame and filter if its on_track or no

- We create a lookup table in ram with the h5 file to organise in this format:
```python
{
  "seq42_img101": {
      "base_name": "nom_final_pour_yolo",
      "annotations": [
          {"class": 0, "x": 100, "y": 200, "w": 50, "h": 50}, # Panneau 1
          {"class": 2, "x": 300, "y": 210, "w": 45, "h": 45}  # Panneau 2 sur la même photo !
      ]
  },
```

- Now we have a lookup table and we open each image and if its in the lookup table we use process_image to put it in the dataset.
By the same time we use multiprocess to do multiple images at once.