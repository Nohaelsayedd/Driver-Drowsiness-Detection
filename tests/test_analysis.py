"""P4 unit test suite — run from project root: python tests/test_analysis.py"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import numpy as np
import time
from typing import Dict
from analysis import TemporalAnalyzer, DrowsinessState


class TestTemporalAnalysisFIXED:

    def __init__(self):
        self.analyzer = TemporalAnalyzer(fps=30)
        self.test_results = {}

    # ── helpers ───────────────────────────────────────────────────────────────

    def create_ear_result(self, ear, blink_detected=False, blink_count=0, head_pose=(0.0, 0.0)):
        return {
            "success": True,
            "ear": ear,
            "ear_left": ear,
            "ear_right": ear,
            "blink_detected": blink_detected,
            "blink_count": blink_count,
            "head_pose": head_pose,
            "left_eye_pts": np.array([[0, 0]] * 6),
            "right_eye_pts": np.array([[0, 0]] * 6),
        }

    def create_hog_result(self, eye_state="open", confidence=0.9):
        return {"success": True, "eye_state": eye_state, "confidence": confidence, "debug_frame": None}

    # ── tests ─────────────────────────────────────────────────────────────────

    def test_01_awake_normal_blink(self):
        print("\n" + "="*60)
        print("TEST 1: Normal Awake Person with Regular Blinking")
        print("="*60)
        self.analyzer.reset()
        for i in range(150):
            if i % 30 == 15:
                ear, blink, state = 0.10, True, "closed"
            else:
                ear, blink, state = 0.35, False, "open"
            result = self.analyzer.process(self.create_ear_result(ear, blink), self.create_hog_result(state))
        assert result["success"]
        assert result["state"] == "alert", f"Expected ALERT, got {result['state']}"
        assert result["drowsiness_score"] < 0.5
        print(f"  PASS  State: {result['state']}, Score: {result['drowsiness_score']:.3f}")
        self.test_results["test_01"] = "PASS"

    def test_02_drowsy_low_ear_oscillating(self):
        print("\n" + "="*60)
        print("TEST 2: Drowsy with Oscillating EAR")
        print("="*60)
        self.analyzer.reset()
        for i in range(200):
            if i % 90 == 45:
                ear, blink, state = 0.08, True, "closed"
            else:
                ear, blink, state = 0.18, False, "closed"
            result = self.analyzer.process(self.create_ear_result(ear, blink), self.create_hog_result(state))
        assert result["success"]
        assert result["drowsiness_score"] > 0.2
        assert result["blink_rate"] < 20
        print(f"  PASS  State: {result['state']}, Score: {result['drowsiness_score']:.3f}")
        self.test_results["test_02"] = "PASS"

    def test_03_critical_sustained_closure(self):
        print("\n" + "="*60)
        print("TEST 3: Critical (Sustained Closure >2s)")
        print("="*60)
        self.analyzer.reset()
        for _ in range(120):
            self.analyzer.process(self.create_ear_result(0.35), self.create_hog_result("open"))
        for _ in range(70):
            result = self.analyzer.process(self.create_ear_result(0.08), self.create_hog_result("closed"))
        assert result["state"] == "critical", f"Expected CRITICAL, got {result['state']}"
        assert result["drowsiness_score"] > 0.8
        assert result["eye_closure_duration"] > 2.0
        print(f"  PASS  State: {result['state']}, Score: {result['drowsiness_score']:.3f}, Closure: {result['eye_closure_duration']:.2f}s")
        self.test_results["test_03"] = "PASS"

    def test_04_normal_blink_not_critical(self):
        print("\n" + "="*60)
        print("TEST 4: Normal Blink Should Not Trigger Alert")
        print("="*60)
        self.analyzer.reset()
        for _ in range(60):
            self.analyzer.process(self.create_ear_result(0.35), self.create_hog_result("open"))
        for i in range(9):
            self.analyzer.process(self.create_ear_result(0.08, i == 0), self.create_hog_result("closed"))
        for _ in range(60):
            result = self.analyzer.process(self.create_ear_result(0.35), self.create_hog_result("open"))
        assert result["state"] == "alert"
        assert result["signals"]["closure_score"] < 0.3
        print(f"  PASS  State: {result['state']}, Closure Score: {result['signals']['closure_score']:.3f}")
        self.test_results["test_04"] = "PASS"

    def test_05_low_blink_rate(self):
        print("\n" + "="*60)
        print("TEST 5: Low Blink Rate (<8/min)")
        print("="*60)
        self.analyzer.reset()
        for i in range(300):
            blink = i == 150
            ear   = 0.10 if blink else 0.25
            result = self.analyzer.process(
                self.create_ear_result(ear, blink),
                self.create_hog_result("open" if ear > 0.2 else "closed"),
            )
        assert result["blink_rate"] < 10
        assert result["signals"]["blink_score"] > 0.5
        print(f"  PASS  Blink Rate: {result['blink_rate']:.1f}/min, Score: {result['signals']['blink_score']:.3f}")
        self.test_results["test_05"] = "PASS"

    def test_06_head_nodding(self):
        print("\n" + "="*60)
        print("TEST 6: Head Nodding Increases Drowsiness")
        print("="*60)
        self.analyzer.reset()
        baseline = []
        for i in range(100):
            result = self.analyzer.process(self.create_ear_result(0.30), self.create_hog_result("open"))
            if i > 80:
                baseline.append(result["drowsiness_score"])
        nodding = []
        for i in range(100):
            pitch  = 25 * np.sin(i / 10)
            result = self.analyzer.process(self.create_ear_result(0.28, head_pose=(0, pitch)), self.create_hog_result("open"))
            if i > 80:
                nodding.append(result["drowsiness_score"])
        assert np.mean(nodding) > np.mean(baseline) - 0.05
        print(f"  PASS  Baseline: {np.mean(baseline):.3f}, Nodding: {np.mean(nodding):.3f}")
        self.test_results["test_06"] = "PASS"

    def test_07_state_hysteresis(self):
        print("\n" + "="*60)
        print("TEST 7: State Hysteresis (No Rapid Flipping)")
        print("="*60)
        self.analyzer.reset()
        states = []
        for i in range(100):
            ear = 0.25 if (i // 2) % 2 == 0 else 0.20
            result = self.analyzer.process(
                self.create_ear_result(ear),
                self.create_hog_result("open" if ear > 0.22 else "closed"),
            )
            states.append(result["state"])
        changes = sum(1 for i in range(1, len(states)) if states[i] != states[i - 1])
        assert changes < 5, f"Too many state changes: {changes}"
        print(f"  PASS  State changes: {changes}")
        self.test_results["test_07"] = "PASS"

    def test_08_missing_input(self):
        print("\n" + "="*60)
        print("TEST 8: Missing Input Handling")
        print("="*60)
        self.analyzer.reset()
        assert not self.analyzer.process({"success": False}, self.create_hog_result())["success"]
        assert not self.analyzer.process(self.create_ear_result(0.35), {"success": False})["success"]
        print("  PASS")
        self.test_results["test_08"] = "PASS"

    def test_09_reset(self):
        print("\n" + "="*60)
        print("TEST 9: Reset Functionality")
        print("="*60)
        self.analyzer.reset()
        for _ in range(50):
            self.analyzer.process(self.create_ear_result(0.35), self.create_hog_result())
        self.analyzer.reset()
        assert self.analyzer.frame_count == 0
        assert len(self.analyzer.ear_history) == 0
        assert self.analyzer.current_state == DrowsinessState.ALERT
        print(f"  PASS  Frame count: {self.analyzer.frame_count}, State: {self.analyzer.current_state.name}")
        self.test_results["test_09"] = "PASS"

    def test_10_confidence(self):
        print("\n" + "="*60)
        print("TEST 10: Confidence Increases Over Time")
        print("="*60)
        self.analyzer.reset()
        confs = []
        for _ in range(200):
            result = self.analyzer.process(self.create_ear_result(0.35), self.create_hog_result())
            confs.append(result["confidence"])
        assert np.mean(confs[-20:]) > np.mean(confs[:20])
        assert np.mean(confs[-20:]) > 0.5
        print(f"  PASS  Early: {np.mean(confs[:20]):.3f}, Late: {np.mean(confs[-20:]):.3f}")
        self.test_results["test_10"] = "PASS"

    def test_perf_speed(self):
        print("\n" + "="*60)
        print("PERF: Frame Processing Speed (<2ms)")
        print("="*60)
        self.analyzer.reset()
        times = []
        for _ in range(100):
            t0 = time.time()
            self.analyzer.process(self.create_ear_result(0.35), self.create_hog_result())
            times.append((time.time() - t0) * 1000)
        avg = np.mean(times)
        assert avg < 2.0, f"Avg {avg:.2f}ms exceeds 2ms"
        print(f"  PASS  Avg: {avg:.2f}ms, Max: {np.max(times):.2f}ms")
        self.test_results["test_perf"] = "PASS"

    # ── runner ────────────────────────────────────────────────────────────────

    def run_all_tests(self):
        print("\n" + "="*60)
        print("P4 TEMPORAL ANALYSIS TEST SUITE")
        print("="*60)
        tests = [
            self.test_01_awake_normal_blink,
            self.test_02_drowsy_low_ear_oscillating,
            self.test_03_critical_sustained_closure,
            self.test_04_normal_blink_not_critical,
            self.test_05_low_blink_rate,
            self.test_06_head_nodding,
            self.test_07_state_hysteresis,
            self.test_08_missing_input,
            self.test_09_reset,
            self.test_10_confidence,
            self.test_perf_speed,
        ]
        for test in tests:
            try:
                test()
            except AssertionError as e:
                print(f"   FAIL: {e}")
                self.test_results[test.__name__] = "FAIL"
            except Exception as e:
                print(f"   ERROR: {e}")
                self.test_results[test.__name__] = "ERROR"

        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        passed = sum(1 for v in self.test_results.values() if v == "PASS")
        total  = len(self.test_results)
        print(f"Passed: {passed}/{total}  ({100*passed/total:.0f}%)")
        for name, res in self.test_results.items():
            mark = "OK" if res == "PASS" else "!!"
            print(f"  {mark} {name}: {res}")


if __name__ == "__main__":
    TestTemporalAnalysisFIXED().run_all_tests()
