import os
import subprocess
import sys


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_FILES = [
    "user_behavior_producer.py",
    "recommendation_algorithms.py",
    "simple_deepfm.py",
    "recommendation_fusion.py",
    "advanced_recommendation_consumer.py",
    "realtime_recommendation_service.py",
    "spark_recommendation_trainer.py",
    "recommendation_consumer.py",
]

JAVA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "flink-jobs", "realtime-recommender-flink-job"))
JAVA_FILES = [
    "src/main/java/com/edu/bigdata/FastRecommender.java",
    "src/main/java/com/edu/bigdata/FeatureExtraction.java",
    "src/main/java/com/edu/bigdata/RealtimeRecommendation.java",
    "pom.xml",
]


def check_file(path: str) -> bool:
    if os.path.exists(path):
        print(f"OK   {path}")
        return True
    print(f"MISS {path}")
    return False


def main() -> int:
    print("=" * 80)
    print("Verify lab 03 code layout")
    print("=" * 80)

    all_ok = True

    print("\nPython services:")
    for filename in PYTHON_FILES:
        all_ok = check_file(os.path.join(BASE_DIR, filename)) and all_ok

    print("\nPython syntax:")
    for filename in ["simple_deepfm.py", "recommendation_algorithms.py"]:
        filepath = os.path.join(BASE_DIR, filename)
        if os.path.exists(filepath):
            result = subprocess.run(
                [sys.executable, "-m", "py_compile", filename],
                cwd=BASE_DIR,
                check=False,
            )
            if result.returncode == 0:
                print(f"OK   {filename}")
            else:
                print(f"FAIL {filename}")
                all_ok = False

    print("\nJava/Flink job:")
    for filepath in JAVA_FILES:
        all_ok = check_file(os.path.join(JAVA_DIR, filepath)) and all_ok

    print("\n" + "=" * 80)
    print("PASS" if all_ok else "FAIL")
    print("=" * 80)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
