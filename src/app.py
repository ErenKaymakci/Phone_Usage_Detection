from services.FrameSourceCV import FrameSourceCV
from services.YOLOPoseEstimator import YOLOPoseEstimator
from services.YOLODetector import YOLODetector
from services.PhoneUsageRule import PhoneUsageRule
from pipeline.InferencePipeline import InferencePipeline

def main():
    reader = FrameSourceCV()

    pose_detector = YOLOPoseEstimator("yolo11n-pose.pt", conf=0.7)
    phone_detector = YOLODetector("yolo12x.pt", conf=0.4, classes=["cell phone", "person"])
    usage_rule = PhoneUsageRule(threshold=150)

    pipeline = InferencePipeline(
        video_reader=reader,
        pose_detector=pose_detector,
        phone_detector=phone_detector,
        usage_rule=usage_rule
    )
    pipeline.run("8873180-hd_1920_1080_25fps.mp4")

if __name__ == "__main__":
    main()
