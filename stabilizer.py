import cv2 as cv
import numpy as np
import math as m
from maritimeReference.FastHorizonAlg import FastHorizon
import os


class VideoStabilizer:
    def __init__(self):
        self.pi = m.pi
        self.ref_detector = FastHorizon()

    def ycord(self, x, rho, theta):
        return (rho - x * np.cos(theta)) / np.sin(theta)

    def rad2deg(self, rad):
        return rad * (180 / self.pi)

    def getTranslationMatrix(self, tx, ty):
        return np.float32([[1, 0, tx], [0, 1, ty]])

    def stabilize(self, input_video):
        output_video = f"stabilized{os.path.splitext(input_video)[0]}.avi"
        cap = cv.VideoCapture(input_video)
        ret, frame = cap.read()
        if not ret:
            print("Error reading video file")
            return

        frame_w, frame_h = frame.shape[1], frame.shape[0]
        x_cen, y_cen = frame_w / 2, frame_h / 2

        fourcc = cv.VideoWriter_fourcc('M', 'J', 'P', 'G')
        vid_stab = cv.VideoWriter(output_video, fourcc, 30.0, (frame_w, frame_h), True)
        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        frame_counter = 0
        while ret:
            frame_no_overlay = frame.copy()
            Y_det, alpha_det, _, _ = self.ref_detector.get_horizon(frame_no_overlay, frame_w, frame_h)

            alpha_ref = 0
            D_alpha = alpha_ref - alpha_det

            Y_ref = y_cen
            D_Y = Y_ref - Y_det

            Rm = cv.getRotationMatrix2D((x_cen, y_cen), D_alpha, 1)
            Tx = self.getTranslationMatrix(0, D_Y)

            I_stab = cv.warpAffine(frame_no_overlay, Rm, (frame_w, frame_h))
            I_stab = cv.warpAffine(I_stab, Tx, (frame_w, frame_h))

            vid_stab.write(I_stab)
            print(f"{frame_counter}/{total_frames}")
            frame_counter += 1

            ret, frame = cap.read()

        cap.release()
        vid_stab.release()
        print("Video stabilization completed")


# Usage example
if __name__ == "__main__":
    stabilizer = VideoStabilizer()
    stabilizer.stabilize('cleaned_MVI_0788_VIS_OB.avi', 'Stabilized.avi')
