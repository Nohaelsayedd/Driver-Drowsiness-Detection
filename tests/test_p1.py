"""
  A) A static image  (no webcam needed)
  B) Live webcam     (press Q to quit)

"""

import argparse
import cv2
import numpy as np
from p1_segmentation import FaceSegmentor


def test_image(image_path: str):
    """Test on a single static image and save output."""
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[ERROR] Could not read image: {image_path}")
        return

    seg = FaceSegmentor()
    result = seg.process(frame)

    if not result["success"]:
        print("[RESULT] No face detected.")
        return

    x, y, w, h = result["bbox"]
    print(f"[RESULT] Face detected!")
    print(f"         BBox      : x={x}, y={y}, w={w}, h={h}")
    print(f"         Crop size : {result['face_crop'].shape}")

    # Save outputs
    cv2.imwrite("output_debug.jpg",    result["debug_frame"])
    cv2.imwrite("output_face_crop.jpg", result["face_crop"])
    cv2.imwrite("output_skin_mask.jpg", result["skin_mask"])

    print("\noutput_debug.jpg: frame with bbox drawn")
    print("output_face_crop.jpg: cropped face")
    print("output_skin_mask.jpg: binary skin mask")

    # Show windows
    cv2.imshow("Debug Frame",  result["debug_frame"])
    cv2.imshow("Face Crop",    result["face_crop"])
    cv2.imshow("Skin Mask",    result["skin_mask"])
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_webcam():
    """Test on live webcam feed. Press Q to quit."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    seg = FaceSegmentor(scale_factor=1.05, min_neighbors=3)
    print("[WEBCAM] Running P1 — press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = seg.process(frame)

        if result["success"]:
            cv2.imshow("P1 - Debug",     result["debug_frame"])
            cv2.imshow("P1 - Face Crop", result["face_crop"])
        else:
            cv2.putText(frame, "No face detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow("P1 - Debug", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test P1 Segmentation")
    parser.add_argument("--mode", choices=["image", "webcam"],
                        default="webcam", help="Test mode")
    parser.add_argument("--path", type=str, default="",
                        help="Path to image (required if mode=image)")
    args = parser.parse_args()

    if args.mode == "image":
        if not args.path:
            print("[ERROR] Provide --path your_image.jpg")
        else:
            test_image(args.path)
    else:
        test_webcam()
