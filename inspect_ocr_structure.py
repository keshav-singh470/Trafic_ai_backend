from paddleocr import PaddleOCR
import cv2
import numpy as np

def inspect_ocr():
    ocr = PaddleOCR(lang='en')
    # Create a dummy image with some text-like noise
    img = np.zeros((100, 300, 3), dtype=np.uint8)
    cv2.putText(img, "TEST1234", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    print("Running OCR...")
    result = ocr.ocr(img)
    print("\n--- RAW RESULT ---")
    print(f"Type: {type(result)}")
    print(f"Content: {result}")
    
    if result:
        print(f"Length of result: {len(result)}")
        for i, res in enumerate(result):
            print(f"\nResult[{i}] Type: {type(res)}")
            print(f"Result[{i}] Content: {res}")
            if res and len(res) > 0:
                line = res[0]
                print(f"  Line[0] Type: {type(line)}")
                print(f"  Line[0] Content: {line}")
                if len(line) > 1:
                    print(f"    Line[0][1] Content: {line[1]}")

if __name__ == "__main__":
    inspect_ocr()
