import cv2

class PhoneUsageRule:
    def __init__(self, threshold: float = 150):
        self.threshold = threshold
        self.phone_in_use = False
        self.use_start_time = None
        self.use_intervals = []

    def check_usage(self, frame, fps, frame_index, persons, phones):
        """
        persons: [{'bbox': [...], 'keypoints': [...]}]
        phones: [{'bbox': [...], 'label': 'cell phone', 'confidence': ...}]
        """
        annotated_frame = frame.copy()
        phone_detected_in_frame = False
        current_time_sec = frame_index / fps

        for person in persons:
            kps = person["keypoints"]
            left_wrist, right_wrist = kps[9], kps[10]

            for phone in phones:
                if phone["label"] != "cell phone":
                    continue
                x1, y1, x2, y2 = map(int, phone["bbox"])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                dist_left = ((left_wrist[0] - cx) ** 2 + (left_wrist[1] - cy) ** 2) ** 0.5
                dist_right = ((right_wrist[0] - cx) ** 2 + (right_wrist[1] - cy) ** 2) ** 0.5
                print("DISTANCE LEFT: ", dist_left, "DISTANCE RIGHT: ", dist_right)

                if dist_left < self.threshold or dist_right < self.threshold:
                    phone_detected_in_frame = True
                    cv2.putText(annotated_frame, "Phone is being used!",
                                (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 255, 0), 3)

                    cv2.circle(annotated_frame, (int(left_wrist[0]), int(left_wrist[1])), 8, (0, 0, 255), -1)  # mavi
                    # Sağ el
                    cv2.circle(annotated_frame, (int(right_wrist[0]), int(right_wrist[1])), 8, (0, 0, 255), -1)  # yeşil
                    # Telefon merkezi
                    cv2.circle(annotated_frame, (cx, cy), 8, (0, 0, 255), -1)  # kırmızı

                    # Sol el → telefon merkezi çizgi
                    cv2.line(annotated_frame, (int(left_wrist[0]), int(left_wrist[1])), (cx, cy), (0, 0, 255), 2)
                    # Sağ el → telefon merkezi çizgi
                    cv2.line(annotated_frame, (int(right_wrist[0]), int(right_wrist[1])), (cx, cy), (0, 0, 255), 2)

                    # Telefon bbox çiz
                    cv2.rectangle(
                        annotated_frame, (x1, y1), (x2, y2),
                        (0, 0, 255), 2
                    )
                    cv2.putText(
                        annotated_frame, f"{phone['label']} {phone['confidence']:.2f}",
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
                    )

        if phone_detected_in_frame and not self.phone_in_use:
            self.use_start_time = current_time_sec
            self.phone_in_use = True
        elif not phone_detected_in_frame and self.phone_in_use:
            self.use_intervals.append((self.use_start_time, current_time_sec))
            self.phone_in_use = False

        return annotated_frame

    def finalize(self, total_frames, fps):
        if self.phone_in_use:
            self.use_intervals.append((self.use_start_time, total_frames / fps))

        total_use_time = sum(end - start for start, end in self.use_intervals)
        return total_use_time, self.use_intervals
