import time
import threading
import queue
import subprocess
import os
import winsound


class InteractionManager:
    def __init__(self):
        self._speaking    = False
        self.cooldowns    = {}
        self.cooldown_duration      = 12.0  # 12 seconds before same key can repeat
        self.global_patience        = 4.0   # 4 seconds between any two alerts
        self.last_global_alert_time = 0
        self.spoken_once  = set()

        self.speech_queue  = queue.Queue(maxsize=3)
        self._ps_process   = None
        self._ps_ready     = False

        # Start persistent PowerShell TTS process
        self._start_ps_server()

        # Worker thread sends queued text to PowerShell stdin
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()

        # Diagnostic beep
        try:
            winsound.Beep(880, 250)
            print("[AUDIO] Startup beep OK.")
        except Exception as e:
            print(f"[AUDIO] Beep failed: {e}")

    def _start_ps_server(self):
        """Launch a persistent PowerShell TTS process."""
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "speak_server.ps1")
        try:
            self._ps_process = subprocess.Popen(
                ["powershell", "-NoProfile", "-NonInteractive",
                 "-ExecutionPolicy", "Bypass", "-File", script_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                bufsize=1,  # line-buffered
            )
            # Wait for the "READY" signal (max 10 seconds)
            ready_thread = threading.Thread(target=self._wait_for_ready, daemon=True)
            ready_thread.start()
            print("[VOICE] PowerShell TTS server starting...")
        except Exception as e:
            print(f"[VOICE] Failed to start TTS server: {e}")

    def _wait_for_ready(self):
        """Read stdout until we see 'READY' from speak_server.ps1."""
        try:
            for line in self._ps_process.stdout:
                line = line.strip()
                if line.startswith("VOICE:"):
                    print(f"[VOICE] Selected voice: {line.replace('VOICE: ', '')}")
                if "READY" in line:
                    self._ps_ready = True
                    print("[VOICE] TTS server is READY — speech will be instant.")
                    break
        except Exception:
            pass

    def _speak(self, text):
        """Send text to the persistent PowerShell process via stdin."""
        if self._ps_process is None or self._ps_process.poll() is not None:
            # Process died — restart it
            print("[VOICE] TTS server died, restarting...")
            self._start_ps_server()
            time.sleep(3)  # give it time to reload

        try:
            self._ps_process.stdin.write(text + "\n")
            self._ps_process.stdin.flush()
        except Exception as e:
            print(f"[VOICE] Speak error: {e}")

    def _worker(self):
        """Worker thread: takes text from queue and sends to TTS."""
        while True:
            try:
                text = self.speech_queue.get(timeout=2)
                if text:
                    self._speaking = True
                    print(f"[VOICE] >>> {text}")
                    self._speak(text)
                    # Estimate speaking time to throttle _speaking flag
                    words = len(text.split())
                    speak_time = max(1.0, words * 0.45)  # ~0.45s per word
                    time.sleep(speak_time)
                    self._speaking = False
                self.speech_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[VOICE] Worker error: {e}")
                self._speaking = False

    def say(self, alert_key, message, once=False, priority=False):
        """
        Queue a voice alert.
        alert_key : unique key to track cooldown
        message   : text to speak
        once      : only speak once per session (reset with reset_once)
        priority  : skip global cooldown (use for DANGER alerts)
        """
        now = time.time()

        if once and alert_key in self.spoken_once:
            return

        if alert_key in self.cooldowns:
            if now - self.cooldowns[alert_key] < self.cooldown_duration:
                return

        if not priority:
            if now - self.last_global_alert_time < self.global_patience:
                return

        if self._speaking:
            return

        try:
            self.speech_queue.put_nowait(message)
            self.cooldowns[alert_key]        = now
            self.last_global_alert_time      = now
            if once:
                self.spoken_once.add(alert_key)
        except queue.Full:
            pass

    def reset_once(self, alert_key):
        """Allow a once-only alert to fire again."""
        self.spoken_once.discard(alert_key)

    def reset_alerts(self, current_labels):
        """Kept for backward compatibility."""
        pass