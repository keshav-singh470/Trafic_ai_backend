try:
    from shapely.geometry import Polygon, Point
    print("Shapely is installed")
except ImportError as e:
    print(f"Shapely is NOT installed: {e}")

try:
    import cv2
    print("OpenCV is installed")
except ImportError as e:
    print(f"OpenCV is NOT installed: {e}")
