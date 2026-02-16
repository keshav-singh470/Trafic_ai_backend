from main import TrafficAnalyticsService
import os

def test():
    video = r"d:\Python\Smart-Traffic-Analytics\test-video\ANPR_BASAVESHWARA_FLY_2.mp4"
    output = "test_output.mp4"
    
    if not os.path.exists(video):
        print(f"Video not found: {video}")
        return

    service = TrafficAnalyticsService(base_model="yolo11l.pt", anpr_model="best.pt")
    print("Processing Case: ANPR...")
    res = service.process_video(video, output, mode="anpr")
    print(f"Results: {res.get('total_violations')} detections")

if __name__ == "__main__":
    test()
