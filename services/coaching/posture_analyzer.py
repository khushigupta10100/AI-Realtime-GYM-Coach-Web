"""Exercise-specific posture analysis for the live trainer.

Uses the existing MediaPipe landmarks and detector metrics. Issues are only
reported when the relevant landmarks are visible with enough confidence.
"""

from __future__ import annotations

import math
import time


NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28

SETUP_PRIORITY = 1
DETECTION_PRIORITY = 2
START_PRIORITY = 3
POSTURE_PRIORITY = 4
PRAISE_PRIORITY = 8

COLOR_GREEN = (70, 190, 80)
COLOR_YELLOW = (0, 200, 255)
COLOR_RED = (40, 50, 230)
COLOR_GRAY = (160, 160, 160)


def _point(landmarks, idx, min_vis=0.7):
    if landmarks is None or idx >= len(landmarks):
        return None
    landmark = landmarks[idx]
    if getattr(landmark, "visibility", 0) < min_vis:
        return None
    return landmark


def _visible(landmarks, indices, min_vis=0.7):
    return all(_point(landmarks, idx, min_vis) is not None for idx in indices)


class PostureAnalyzer:
    def __init__(self):
        self._prev_centroid = None
        self._prev_centroid_at = 0.0
        self._had_issue = False
        self.mistake_counts = {}
        self.correct_reps = 0
        self.incorrect_attempts = 0
        self._last_counted_reps = 0
        self._phase_had_blocking = False

    def reset(self):
        self._prev_centroid = None
        self._prev_centroid_at = 0.0
        self._had_issue = False
        self.mistake_counts = {}
        self.correct_reps = 0
        self.incorrect_attempts = 0
        self._last_counted_reps = 0
        self._phase_had_blocking = False

    def gate_repetition(self, detector, metrics, blocking):
        """Keep detector counting, but accept a rep only when form is acceptable."""
        current = int(getattr(detector, "reps", 0) or 0)
        if current > self._last_counted_reps:
            delta = current - self._last_counted_reps
            if blocking or self._phase_had_blocking:
                detector.reps = self._last_counted_reps
                self.incorrect_attempts += delta
                metrics["reps"] = detector.reps
            else:
                self.correct_reps += delta
                self._last_counted_reps = detector.reps
            self._phase_had_blocking = False
        elif blocking:
            self._phase_had_blocking = True
        metrics["correct_reps"] = self.correct_reps
        metrics["incorrect_attempts"] = self.incorrect_attempts
        return metrics

    def analyze(self, exercise, landmarks, metrics):
        """Return coaching analysis merged with detector metrics."""
        metrics = dict(metrics or {})
        issues = []

        if not landmarks:
            issues.append(self._issue(
                "no_pose",
                DETECTION_PRIORITY,
                "error",
                "Please stand in front of the camera.",
                "I cannot see your body yet. Stand in front of the camera.",
                [],
                blocking=True,
            ))
            return self._pack(exercise, metrics, issues, landmarks)

        framing_issue = self._framing_issue(landmarks, exercise)
        if framing_issue:
            issues.append(framing_issue)
            return self._pack(exercise, metrics, issues, landmarks)

        facing_issue = self._facing_issue(landmarks)
        if facing_issue:
            issues.append(facing_issue)

        start_issue = self._starting_position_issue(exercise, landmarks, metrics)
        if start_issue:
            issues.append(start_issue)

        issues.extend(self._exercise_issues(exercise, landmarks, metrics))

        excessive = self._excessive_movement_issue(landmarks, exercise)
        if excessive:
            issues.append(excessive)

        return self._pack(exercise, metrics, issues, landmarks)

    def register_rep_attempt(self, current_reps, blocking):
        """Track correct vs incorrect attempts when the detector counts a new rep."""
        if current_reps > self._last_counted_reps:
            added = current_reps - self._last_counted_reps
            if blocking or self._phase_had_blocking:
                self.incorrect_attempts += added
                accepted = False
            else:
                self.correct_reps += added
                accepted = True
            self._last_counted_reps = current_reps
            self._phase_had_blocking = False
            return accepted
        if blocking:
            self._phase_had_blocking = True
        return True

    def _issue(self, code, priority, severity, short_label, cue, joints, blocking=False):
        return {
            "code": code,
            "priority": priority,
            "severity": severity,
            "label": short_label,
            "cue": cue,
            "joints": joints,
            "blocking": blocking,
        }

    def _framing_issue(self, landmarks, exercise):
        required = {
            "Squats": [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE],
            "Lunges": [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE],
            "Push-ups": [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW, LEFT_HIP, RIGHT_HIP, LEFT_ANKLE, RIGHT_ANKLE],
            "Biceps Curls (Dumbbell)": [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST],
            "Shoulder Press": [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP],
        }.get(exercise, [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP])

        missing_lower = exercise in ("Squats", "Lunges", "Push-ups") and not _visible(
            landmarks, [LEFT_ANKLE, RIGHT_ANKLE], 0.55
        )
        if missing_lower:
            return self._issue(
                "full_body",
                SETUP_PRIORITY,
                "warning",
                "Full body not visible",
                "Move slightly backward so your full body is visible.",
                [LEFT_ANKLE, RIGHT_ANKLE],
                blocking=True,
            )

        if not _visible(landmarks, required, 0.55):
            return self._issue(
                "low_confidence",
                DETECTION_PRIORITY,
                "warning",
                "Body not fully detected",
                "Adjust your position so I can detect your body properly.",
                required[:4],
                blocking=True,
            )

        visible_points = [landmarks[i] for i in required if _point(landmarks, i, 0.5)]
        if not visible_points:
            return self._issue(
                "no_pose",
                DETECTION_PRIORITY,
                "error",
                "Please stand in front of the camera.",
                "Please stand in front of the camera.",
                [],
                blocking=True,
            )

        xs = [p.x for p in visible_points]
        ys = [p.y for p in visible_points]
        if min(xs) < 0.03 or max(xs) > 0.97 or min(ys) < 0.02 or max(ys) > 0.98:
            return self._issue(
                "out_of_frame",
                SETUP_PRIORITY,
                "warning",
                "Stay inside the frame",
                "Keep your entire body inside the camera frame.",
                [],
                blocking=True,
            )
        return None

    def _facing_issue(self, landmarks):
        left = _point(landmarks, LEFT_SHOULDER, 0.55)
        right = _point(landmarks, RIGHT_SHOULDER, 0.55)
        nose = _point(landmarks, NOSE, 0.4)
        if left is None or right is None:
            return self._issue(
                "face_camera",
                SETUP_PRIORITY,
                "warning",
                "Face the camera",
                "Face the camera so I can see both shoulders.",
                [LEFT_SHOULDER, RIGHT_SHOULDER],
                blocking=True,
            )
        width = abs(left.x - right.x)
        if width < 0.10:
            return self._issue(
                "face_camera",
                SETUP_PRIORITY,
                "warning",
                "Face the camera",
                "Face the camera. Turn toward me so I can see your full posture.",
                [LEFT_SHOULDER, RIGHT_SHOULDER],
                blocking=True,
            )
        if nose is None and width < 0.16:
            return self._issue(
                "face_camera",
                SETUP_PRIORITY,
                "warning",
                "Face the camera",
                "Face the camera.",
                [NOSE],
                blocking=False,
            )
        return None

    def _starting_position_issue(self, exercise, landmarks, metrics):
        if metrics.get("reps", 0) > 0:
            return None
        if exercise == "Squats":
            knee = metrics.get("knee_angle", 180)
            if isinstance(knee, (int, float)) and 70 < knee < 140:
                return None
            if isinstance(knee, (int, float)) and knee >= 155:
                return self._issue(
                    "start_squat",
                    START_PRIORITY,
                    "ok",
                    "Ready to squat",
                    "Stand tall, feet about shoulder-width apart, then begin your squat.",
                    [LEFT_KNEE, RIGHT_KNEE],
                )
        if exercise == "Push-ups":
            elbow = metrics.get("elbow_angle", 180)
            if isinstance(elbow, (int, float)) and elbow >= 150:
                alignment = metrics.get("body_alignment", "")
                if alignment == "Poor Form":
                    return self._issue(
                        "start_pushup",
                        START_PRIORITY,
                        "warning",
                        "Set your plank",
                        "Set a straight plank from head to heels before you lower down.",
                        [LEFT_HIP, RIGHT_HIP],
                    )
        return None

    def _exercise_issues(self, exercise, landmarks, metrics):
        if exercise == "Squats":
            return self._squat_issues(landmarks, metrics)
        if exercise == "Push-ups":
            return self._pushup_issues(landmarks, metrics)
        if exercise == "Biceps Curls (Dumbbell)":
            return self._curl_issues(landmarks, metrics)
        if exercise == "Shoulder Press":
            return self._press_issues(landmarks, metrics)
        if exercise == "Lunges":
            return self._lunge_issues(landmarks, metrics)
        return []

    def _squat_issues(self, landmarks, metrics):
        issues = []
        knee = metrics.get("knee_angle")
        back = metrics.get("back_angle")
        depth = metrics.get("depth_status", "")

        if isinstance(back, (int, float)) and _visible(landmarks, [LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE], 0.6):
            if back < 125:
                issues.append(self._issue(
                    "squat_back",
                    POSTURE_PRIORITY,
                    "error",
                    "Back is bending",
                    "Your back is bending. Maintain a neutral spine and keep your chest up.",
                    [LEFT_SHOULDER, LEFT_HIP, RIGHT_HIP],
                    blocking=True,
                ))
            elif back < 140:
                issues.append(self._issue(
                    "squat_lean",
                    POSTURE_PRIORITY,
                    "warning",
                    "Leaning forward",
                    "Do not lean forward. Keep your chest tall and sit your hips back.",
                    [LEFT_SHOULDER, LEFT_HIP],
                ))

        if isinstance(knee, (int, float)) and 105 < knee < 145:
            issues.append(self._issue(
                "squat_shallow",
                POSTURE_PRIORITY,
                "warning",
                "Squat too shallow",
                "Your squat is too shallow. Lower your hips a little more.",
                [LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE],
            ))

        valgus = self._knee_valgus(landmarks)
        if valgus:
            issues.append(valgus)

        feet = self._foot_width_issue(landmarks)
        if feet:
            issues.append(feet)

        return issues

    def _knee_valgus(self, landmarks):
        left_hip = _point(landmarks, LEFT_HIP)
        right_hip = _point(landmarks, RIGHT_HIP)
        left_knee = _point(landmarks, LEFT_KNEE)
        right_knee = _point(landmarks, RIGHT_KNEE)
        left_ankle = _point(landmarks, LEFT_ANKLE)
        right_ankle = _point(landmarks, RIGHT_ANKLE)
        if not all([left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle]):
            return None
        mid = (left_hip.x + right_hip.x) / 2.0
        left_in = (left_knee.x - mid) < (left_ankle.x - mid) * 0.65 and abs(left_knee.x - mid) < abs(left_ankle.x - mid) - 0.03
        right_in = (mid - right_knee.x) < (mid - right_ankle.x) * 0.65 and abs(right_knee.x - mid) < abs(right_ankle.x - mid) - 0.03
        if left_in or right_in:
            return self._issue(
                "knees_in",
                POSTURE_PRIORITY,
                "error",
                "Knees moving inward",
                "Your knees are moving inward. Keep them aligned with your toes.",
                [LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE],
                blocking=True,
            )
        return None

    def _foot_width_issue(self, landmarks):
        left_hip = _point(landmarks, LEFT_HIP)
        right_hip = _point(landmarks, RIGHT_HIP)
        left_ankle = _point(landmarks, LEFT_ANKLE)
        right_ankle = _point(landmarks, RIGHT_ANKLE)
        if not all([left_hip, right_hip, left_ankle, right_ankle]):
            return None
        hip_width = abs(left_hip.x - right_hip.x)
        foot_width = abs(left_ankle.x - right_ankle.x)
        if hip_width < 0.04:
            return None
        if foot_width < hip_width * 0.7:
            return self._issue(
                "feet_narrow",
                POSTURE_PRIORITY,
                "warning",
                "Feet too close",
                "Set your feet about shoulder-width apart before you squat.",
                [LEFT_ANKLE, RIGHT_ANKLE],
            )
        return None

    def _pushup_issues(self, landmarks, metrics):
        issues = []
        alignment = metrics.get("body_alignment", "")
        hip_status = metrics.get("hip_status", "")
        elbow = metrics.get("elbow_angle")

        if alignment == "Poor Form":
            issues.append(self._issue(
                "pushup_body",
                POSTURE_PRIORITY,
                "error",
                "Body not aligned",
                "Keep your body in a straight line from shoulders to heels.",
                [LEFT_SHOULDER, LEFT_HIP, LEFT_ANKLE],
                blocking=True,
            ))
        elif alignment == "Slight Bend":
            issues.append(self._issue(
                "pushup_bend",
                POSTURE_PRIORITY,
                "warning",
                "Slight body bend",
                "Keep your back position steady and brace your core.",
                [LEFT_HIP, RIGHT_HIP],
            ))

        if hip_status == "SAGGING":
            issues.append(self._issue(
                "pushup_sag",
                POSTURE_PRIORITY,
                "error",
                "Hips sagging",
                "Your hips are sagging. Squeeze your glutes and keep a strong plank.",
                [LEFT_HIP, RIGHT_HIP],
                blocking=True,
            ))
        elif hip_status == "PIKED UP":
            issues.append(self._issue(
                "pushup_pike",
                POSTURE_PRIORITY,
                "warning",
                "Hips too high",
                "Lower your hips so your body stays in one straight line.",
                [LEFT_HIP, RIGHT_HIP],
            ))

        if isinstance(elbow, (int, float)) and 100 < elbow < 145:
            issues.append(self._issue(
                "pushup_shallow",
                POSTURE_PRIORITY,
                "warning",
                "Incomplete push-up",
                "Lower your chest a little more, then press up to complete the movement.",
                [LEFT_ELBOW, RIGHT_ELBOW],
            ))

        flare = self._elbow_flare(landmarks)
        if flare:
            issues.append(flare)
        return issues

    def _elbow_flare(self, landmarks):
        l_sh = _point(landmarks, LEFT_SHOULDER)
        r_sh = _point(landmarks, RIGHT_SHOULDER)
        l_el = _point(landmarks, LEFT_ELBOW)
        r_el = _point(landmarks, RIGHT_ELBOW)
        if not all([l_sh, r_sh, l_el, r_el]):
            return None
        shoulder_width = abs(l_sh.x - r_sh.x)
        elbow_width = abs(l_el.x - r_el.x)
        if shoulder_width > 0.05 and elbow_width > shoulder_width * 1.85:
            return self._issue(
                "elbow_flare",
                POSTURE_PRIORITY,
                "warning",
                "Elbows flared",
                "Your elbow position is incorrect. Keep your elbows closer to your sides.",
                [LEFT_ELBOW, RIGHT_ELBOW],
            )
        return None

    def _curl_issues(self, landmarks, metrics):
        issues = []
        swing = metrics.get("swing_status", "")
        shoulder = metrics.get("shoulder_status", "")
        elbow = metrics.get("elbow_angle")

        if swing == "SWINGING":
            issues.append(self._issue(
                "curl_swing",
                POSTURE_PRIORITY,
                "error",
                "Body swinging",
                "Avoid unnecessary body movement. Keep your torso still and curl with your arms.",
                [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP],
                blocking=True,
            ))

        if shoulder == "ELBOW DRIFTING":
            issues.append(self._issue(
                "curl_elbow",
                POSTURE_PRIORITY,
                "warning",
                "Elbow drifting",
                "Keep your elbows close to your sides. Do not let them travel forward.",
                [LEFT_ELBOW, RIGHT_ELBOW],
            ))

        if isinstance(elbow, (int, float)) and 70 < elbow < 130:
            issues.append(self._issue(
                "curl_rom",
                POSTURE_PRIORITY,
                "warning",
                "Incomplete curl",
                "Complete the movement before starting another repetition.",
                [LEFT_WRIST, LEFT_ELBOW],
            ))
        return issues

    def _press_issues(self, landmarks, metrics):
        issues = []
        arch = metrics.get("back_arch_status", "")
        extension = metrics.get("extension_status", "")

        if arch == "Excessive Arch":
            issues.append(self._issue(
                "press_arch",
                POSTURE_PRIORITY,
                "error",
                "Lower back arching",
                "Your back is arching. Brace your core and keep a more upright torso.",
                [LEFT_SHOULDER, LEFT_HIP, RIGHT_HIP],
                blocking=True,
            ))
        elif arch == "Slight Arch":
            issues.append(self._issue(
                "press_slight_arch",
                POSTURE_PRIORITY,
                "warning",
                "Slight back arch",
                "Keep your ribs down and your core tight as you press.",
                [LEFT_HIP, RIGHT_HIP],
            ))

        if extension == "NEARLY EXTENDED":
            issues.append(self._issue(
                "press_lockout",
                POSTURE_PRIORITY,
                "warning",
                "Press not finished",
                "Straighten your arms fully at the top, without shrugging your shoulders.",
                [LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST],
            ))
        return issues

    def _lunge_issues(self, landmarks, metrics):
        issues = []
        balance = metrics.get("balance_status", "")
        torso = metrics.get("torso_angle")
        front_knee = metrics.get("front_knee_angle")

        if balance == "OFF BALANCE":
            issues.append(self._issue(
                "lunge_balance",
                POSTURE_PRIORITY,
                "error",
                "Off balance",
                "You are leaning too much. Keep your torso stacked over your hips.",
                [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP],
                blocking=True,
            ))

        if isinstance(torso, (int, float)) and torso < 140:
            issues.append(self._issue(
                "lunge_torso",
                POSTURE_PRIORITY,
                "warning",
                "Torso folding",
                "Keep your back straight and your chest up during the lunge.",
                [LEFT_SHOULDER, LEFT_HIP],
            ))

        if isinstance(front_knee, (int, float)) and 105 < front_knee < 150:
            issues.append(self._issue(
                "lunge_shallow",
                POSTURE_PRIORITY,
                "warning",
                "Lunge too shallow",
                "Lower a little more until your front thigh is closer to parallel.",
                [LEFT_KNEE, RIGHT_KNEE],
            ))
        return issues

    def _excessive_movement_issue(self, landmarks, exercise):
        if exercise not in ("Biceps Curls (Dumbbell)", "Shoulder Press"):
            return None
        left = _point(landmarks, LEFT_HIP, 0.5)
        right = _point(landmarks, RIGHT_HIP, 0.5)
        if left is None or right is None:
            return None
        centroid = ((left.x + right.x) / 2.0, (left.y + right.y) / 2.0)
        now = time.time()
        issue = None
        if self._prev_centroid is not None:
            dt = max(now - self._prev_centroid_at, 1e-3)
            dist = math.hypot(centroid[0] - self._prev_centroid[0], centroid[1] - self._prev_centroid[1])
            speed = dist / dt
            if speed > 0.55:
                issue = self._issue(
                    "excess_move",
                    POSTURE_PRIORITY,
                    "warning",
                    "Moving too much",
                    "Avoid unnecessary body movement. Plant your feet and isolate the lift.",
                    [LEFT_HIP, RIGHT_HIP],
                )
        self._prev_centroid = centroid
        self._prev_centroid_at = now
        return issue

    def _pack(self, exercise, metrics, issues, landmarks):
        issues = [item for item in issues if item]
        issues.sort(key=lambda item: (item["priority"], 0 if item["severity"] == "error" else 1))
        primary = None
        for item in issues:
            if item["severity"] in ("error", "warning"):
                primary = item
                break
        if primary is None and issues:
            primary = issues[0]

        blocking = any(item.get("blocking") for item in issues if item["severity"] in ("error", "warning"))

        status = "correct"
        color = COLOR_GREEN
        if not landmarks:
            status = "no_pose"
            color = COLOR_GRAY
        elif primary and primary["severity"] == "error":
            status = "incorrect"
            color = COLOR_RED
        elif primary and primary["severity"] == "warning":
            status = "warning"
            color = COLOR_YELLOW
        elif primary and primary["priority"] <= START_PRIORITY and primary["severity"] == "ok":
            status = "ready"
            color = COLOR_GREEN

        if primary and primary["severity"] in ("error", "warning"):
            self.mistake_counts[primary["code"]] = self.mistake_counts.get(primary["code"], 0) + 1
            self._had_issue = True
            improved = False
        else:
            improved = self._had_issue
            self._had_issue = False

        instruction = ""
        if primary:
            instruction = primary["cue"]
        elif improved:
            instruction = "Excellent posture! Keep going."
        elif int(metrics.get("reps") or 0) > 0:
            instruction = "Great job! Keep going."

        errors = [item["label"] for item in issues if item["severity"] in ("error", "warning")]
        highlight = []
        if primary:
            highlight = primary.get("joints") or []

        metrics.update({
            "pose_detected": bool(landmarks),
            "posture_status": status,
            "posture_errors": errors,
            "coach_instruction": instruction,
            "coach_status": self._status_label(status, primary),
            "highlight_joints": highlight,
            "form_ok": status in ("correct", "ready") and not blocking,
            "blocking_form": blocking,
            "priority_issue": primary,
            "improved": improved,
            "mistake_counts": dict(self.mistake_counts),
            "overlay_color": color,
        })
        return metrics

    def _status_label(self, status, primary):
        if status == "no_pose":
            return "Waiting for body detection"
        if status == "incorrect":
            return "Correcting form"
        if status == "warning":
            return "Adjustment needed"
        if status == "ready":
            return "Ready — begin the movement"
        if primary and primary["code"].startswith("start"):
            return "Set your starting position"
        return "Good form"
