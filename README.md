# ReflectiveTrack
An object tracker that attempts to filter out moving reflections.

Based on the lk_track sample from OpenCV 3.1.0.

Technique:
* Lucas-Kanade optical flow, initialized with "Good Features to Track".
* Bounding box drawn for moving vehicles.

Usage:
* lk_track.py [<video_source>]
* Press <Esc> to exit.

