import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from pipeline import DrowsinessDetectionPipeline

if __name__ == "__main__":
    pipeline = DrowsinessDetectionPipeline(fps=30)
    pipeline.run()
