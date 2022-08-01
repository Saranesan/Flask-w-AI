from flask import Flask
from Chat_Bot.Chat_Bot import *
from Speech_Conversion.Speech_Util import *
import nltk
nltk.download('wordnet', download_dir='./nltk_data/')
nltk.download('omw-1.4',download_dir='./nltk_data/')

app = Flask(__name__)

@app.route('/')
def index():
	return "Hello, World!"


@app.route('/interview', methods=['POST','GET'])
def interview():
	#Ingress
	URL = getURL()
	downloadFile(URL)
	input = speech_to_text("input.wav")
	output = interview_response(input)

	 #Egress
	text_to_speech(output)
	uploadWav("output.wav")
	return output


@app.route('/resume_grader')
def resume():
	return "yikai-script"

@app.route('/iot_camera')
def iot_camera():
	return "brian-script"


if __name__ == "__main__":
	app.run(port=8080, debug=True)