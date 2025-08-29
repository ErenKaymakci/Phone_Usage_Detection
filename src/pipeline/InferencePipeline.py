import cv2

class InferencePipeline:
    def __init__(self, video_reader, pose_detector, phone_detector, usage_rule):
        self.video_reader = video_reader
        self.pose_detector = pose_detector
        self.phone_detector = phone_detector
        self.usage_rule = usage_rule

    def run(self, video_path):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter("output_annotated.mp4", fourcc, fps, (width, height))

        frame_index = 0
        annotated_frame = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_index += 1

            persons = self.pose_detector.detect(frame)
            phones = self.phone_detector.detect(frame, rois=persons)
            
            annotated_frame = self.usage_rule.check_usage(frame, fps, frame_index, persons, phones)
            out.write(annotated_frame)

        # finalize rapor
        total_use_time, intervals = self.usage_rule.finalize(frame_index, fps)
        print("Telefon kullanım aralıkları:", intervals)
        print("Toplam kullanım süresi:", total_use_time)

        # Son frame üzerine yaz
        if annotated_frame is not None:
            text = f"Total phone usage: {total_use_time:.2f} sec"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
            center_x, center_y = width // 2, height // 2
            org = (center_x - tw // 2, center_y + th // 2)
            cv2.putText(annotated_frame, text, org,
                        cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)

            for _ in range(int(fps * 2)):
                out.write(annotated_frame)

        cap.release()
        out.release()
