#!/usr/bin/env python

'''
Lucas-Kanade optical flow, initialized with "Good Features to Track".
Bounding box drawn for moving vehicles.
Modified from OpenCV 3.1.0.

Usage:
lk_track.py [<video_source>]

Press <Esc> to exit.
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2
import video
from common import anorm2, draw_str
from time import clock
import pdb

lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

class App:
    def __init__(self, video_src):
        self.track_len = 10
        self.detect_interval = 5
        self.tracks = []
        self.cam = video.create_capture(video_src)
        self.frame_idx = 0

    def run(self):
        # corners of the bounding box
        topleft_corner = [0, 0]
        bottomright_corner = [0, 0]
        bbox_centers = []
        smoothing_window = 3

        while True:
            ret, frame = self.cam.read()

            # crop frame to exclude unused background regions
            frame = frame[500:, 700:, :]

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            vis = frame.copy()
            bbox_img = frame.copy()

            if len(self.tracks) > 0:
                img0, img1 = self.prev_gray, frame_gray

                # Initial features
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)

                # Consider those features good which have 
                # similar optical flow going forward and backward in time
                p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                d = abs(p0-p0r).reshape(-1, 2).max(-1)
                good = d < 1

                new_tracks = []
                bbox_points = []

                # tracks has the format [track1, track2,...] where 
                # each track has the format [(x0, y0), (x1, y1),...]
                for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                    if not good_flag:
                        continue
                    tr.append((x, y))
                    
                    if len(tr) > self.track_len:
                        del tr[0]

                    new_tracks.append(tr)
                    cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)

                    if (abs(tr[-1][0] - tr[0][0]) > 10) or (abs(tr[-1][1] - tr[0][1]) > 10):
                        # features that have long tracks behind them
                        # are included in the bounding box
                        bbox_points.append((x,y))
                self.tracks = new_tracks

                # draw multiple tracks at once
                cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))

                draw_str(vis, (20, 20), 'track count: %d' % len(self.tracks))

                if len(bbox_points) > 1:
                    topleft_corner[0] = min([i[0] for i in bbox_points])
                    topleft_corner[1] = min([i[1] for i in bbox_points])
                    bottomright_corner[0] = max([i[0] for i in bbox_points])
                    bottomright_corner[1] = max([i[1] for i in bbox_points])

                    # calculate the center of each bbox for drawing the trajectory
                    new_center = [topleft_corner[0] + 0.5*(bottomright_corner[0] - topleft_corner[0]),
                        topleft_corner[1] + 0.5*(bottomright_corner[1] - topleft_corner[1])]

                    # smooth the trajectory
                    if len(bbox_centers) > smoothing_window:
                        '''
                        # start a new trajectory if trajectory has a large discontinuity
                        if (new_center[0] - bbox_centers[-1][0] > 20) or (new_center[1] - bbox_centers[-1][1] > 20):
                            bbox_centers = []
                            continue

                        if self.frame_idx % 550 == 0:
                            bbox_centers = []
                            continue
                        '''

                        new_center[0] = (1.0/smoothing_window)*abs(new_center[0] + bbox_centers[-1][0] + bbox_centers[-2][0])
                        new_center[1] = (1.0/smoothing_window)*abs(new_center[1] + bbox_centers[-1][1] + bbox_centers[-2][1])

                    bbox_centers.append(tuple(new_center))

                    # draw bbox
                    cv2.rectangle(vis, tuple(topleft_corner), tuple(bottomright_corner), (255, 150, 150), thickness = 3)

                # draw trajectory
                cv2.polylines(vis, [np.int32(bbox_centers)], False, (255, 150, 150), thickness = 3)

            # when frame index is a multiple of the detection interval
            # recalculate good features and 
            # append them as the initial points of new tracks
            if self.frame_idx % self.detect_interval == 0:
                # mask is the region of interest for goodFeaturesToTrack
                mask = np.zeros_like(frame_gray)
                mask[:] = 255

                for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                    cv2.circle(mask, (x, y), 5, 0, -1)

                p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        self.tracks.append([(x, y)])


            self.frame_idx += 1
            self.prev_gray = frame_gray
            cv2.imshow('lk_track', vis)

            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                break

def main():
    import sys
    try:
        video_src = sys.argv[1]
    except:
        video_src = 0

    print(__doc__)
    App(video_src).run()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
