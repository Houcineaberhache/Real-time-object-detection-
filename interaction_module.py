import pyttsx3
import time
import cv2
import threading

class InteractionManager:
    def __init__(self):
        self.last_alert_time = 0
        self.cooldown = 5  # Seconds between alerts
        self._speaking = False  # Guard against overlapping calls

    def _speak(self, text):
        """Run TTS in a separate thread to avoid blocking the video loop."""
        self._speaking = True
        try:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception as e:
            print(f"[TTS Error] {e}")
        finally:
            self._speaking = False

    def trigger_voice_alert(self, label):
        current_time = time.time()
        # Only fire if cooldown has passed AND no alert is currently playing
        if current_time - self.last_alert_time > self.cooldown and not self._speaking:
            self.last_alert_time = current_time
            thread = threading.Thread(target=self._speak, args=(f"{label} detected",), daemon=True)
            thread.start()

    def draw_dashboard(self, frame, fps, count):
        # Professional semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (250, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Signs: {count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        return frame