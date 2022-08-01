import urllib.request
import speech_recognition as sr
import pyttsx3
from dotenv import load_dotenv
load_dotenv()
import cloudinary
import cloudinary.uploader
import cloudinary.api
import json


config = cloudinary.config(secure=True)
r = sr.Recognizer()


#DOWNLOAD CLOUDINARY FILE TO LOCAL/SERVER
def downloadFile(url):
	urllib.request.urlretrieve(url,"input.wav")


def text_to_speech(command):		
	engine = pyttsx3.init()
	engine.save_to_file(command, "output.wav")
	engine.runAndWait()	
	
def speech_to_text(filename):
	with sr.AudioFile(filename) as source:	    
	    audio_data = r.record(source)	    
	    text = r.recognize_google(audio_data)
	    return text

#UPLOAD TO CLOUDINARY
def uploadWav(filename):
  cloudinary.uploader.upload(filename, public_id="input", unique_filename = False, overwrite=True, resource_type = "video")
  srcURL = cloudinary.CloudinaryImage("output").build_url()

#GET CLOUDINARY URL
def getURL():
  image_info=cloudinary.api.resource("input",resource_type="video")
  print("Upload response:\n", json.dumps(image_info,indent=2), "\n")
  return(image_info["url"])





