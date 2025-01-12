import speech_recognition as sr 
import pyttsx3


engine = pyttsx3.init()


def output_audio(text):
    engine.setProperty('rate', 150) 
    engine.setProperty('volume', 1) 

    engine.say(text)
    engine.runAndWait()

output_audio("Hello World")