# Intrusion Detection System

Solution for the project work in Image Processing and Computer Vision, MSc in Artificial Intelligence at Unibo.

The aim is the development of an intrusion detection system able to perceive movement with respect to a static background and perform basic classifications tasks on the found intruders.

Input Example             |  Output Example
:-------------------------:|:-------------------------:
![GIF](https://github.com/alessandrostockman/cv-intrusion-detection-project-work/blob/master/res/input-example.gif)  |  ![GIF](https://github.com/alessandrostockman/cv-intrusion-detection-project-work/blob/master/res/output-example.gif)

## Requirements

```
python==3.9.2
matplotlib==3.4.1
numpy==1.20.2
opencv-python==4.5.1.48
```

## Usage

`python main.py [-I/--input INPUT] [-o/--output OUTPUT_DIRECTORY] [-S/--stats] [-T/--tuning] [-P/--preset PRESET]`

- `--input`: Input video used to compute the intrusion detection algorithm
- `--ouput`: Output directory where the requested output are stored
- `--stats`: Compute and print additional info on the elaborated data
- `--tuning`: Activates tuning mode, in which all the algorithm steps are generated as output videos
- `--preset`: Preset of parameters used from most accurate (1) to fastest (3)
