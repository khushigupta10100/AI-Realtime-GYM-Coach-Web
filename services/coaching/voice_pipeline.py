import time
import streamlit as st
class VoicePipeline:
    def __init__(self, llm, tts):
        self.llm = llm
        self.tts = tts
        self.last_spoken_at = 0

    def _find_form_issue(self, exercise, metrics):
        # Keep your complete existing _find_form_issue code
        pass

    def process_event(self, event, exercise, metrics):
        now = time.time()

        # Handle missing pose without calling the LLM
        if event == "no_pose_detected":
            if now - self.last_spoken_at < 5:
                return None

            text = "Please step into the camera frame."
            voice = self.tts.speak(text)

            self.last_spoken_at = now
            return voice, text

        # Existing logic
        issue = self._find_form_issue(exercise, metrics)

        is_major_issue = event in [
            "workout_started",
            "set_completed",
            "workout_completed"
        ]

        if not is_major_issue:
            if not issue:
                return None

            if now - self.last_spoken_at < 5:
                return None

        text = self.llm.give_feedback(event, issue)
        print("LLM Response:", text)

        voice = self.tts.speak(text)
        print("Voice Bytes:", len(voice) if voice else 0)

        self.last_spoken_at = now
        return voice, text


def autoplay_audio(audio_bytes):
    if not audio_bytes:
        return

    st.markdown(
        "<style>[data-testid='stAudio'] {display: none;}</style>",
        unsafe_allow_html=True
    )
    st.audio(audio_bytes, format="audio/mp3", autoplay=True)
