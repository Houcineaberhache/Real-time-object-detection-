import pyttsx3

engine = pyttsx3.init()
print("Testing voice alert...")
engine.say("System active. Person detected.")
engine.runAndWait()
print("Success!")