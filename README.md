# Flask w AI
 AI Interview Chat Bot on Flask Server

Everything is done on Python 3.7.13


Dependencies

Flask 2.1.3 (Flask server)

NLTK 3.7 (Requires wordnet, omw-1.4, punkt)

SpeechRecognition 3.8.1 ( For conversion of speech to text & text to speech )

pyttsx3 2.71 ( For conversion of speech to text & text to speech )

cloudinary 1.29.0 (Pulling and Posting files to cloudinary)

python-dotenv (For cloudinary environment files)

urllib3 1.26.11 (Downloading Cloudinary Files to server/locally)


gunicorn ( Only if using Google Cloud Platform to host Flask )

numpy 1.21.6 ( General Data manipulation )

AI
torch ( Used for creating neural network, as well as training and predicting )


Folder
Flask -> AI converted into a WEB API through using flask Server. (Routes can be seen through main.py)
Chat_Bot_Training -> Run train.py to produce training data (Introduction.pth)

