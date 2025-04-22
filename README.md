# TrueTone: Precision Matching for your Unique Skin
TrueTone is a computer vision-powered solution designed to help users find their perfect foundation shade using real-time face detection and color analysis. Built with machine learning and YOLOv5, this project provides personalized recommendations from a curated foundation dataset—bridging the gap between digital shopping and personalized beauty.

## 💡 Why TrueTone?
Finding the right foundation online is a frustrating process filled with trial, error, and wasted products. 
TrueTone solves this problem by:
* Capturing your skin tone via webcam
* Analyzing your skin's undertones using color clustering
* Matching it with the closest foundation shades from top beauty retailers
* Providing a smooth and user-friendly GUI experience

## ⚙️ Features
### 🧠 Computer Vision & ML
* YOLOv5-based real-time face detection
* Skin color segmentation and clustering
* KMeans clustering for dominant tone extraction
* Foundation matching using Euclidean distance
### 🖥️ Real-Time GUI
* Live webcam feed
* Face detection bounding box
* Top foundation shade matches displayed instantly
###📦 Foundation Dataset
* Curated CSV dataset of foundation shades
* Contains product names, brands, and RGB color values
* Built for easy scalability across retailers
### 🗂 Project Structure
TrueTone/
├── final_code.py               # Main application logic
├── Final_Foundation_dataset.csv # Curated foundation shade dataset
├── README.md                   # Documentation
└── requirements.txt            # Dependencies (create this with pip freeze)
###🧪 Tech Stack
* Python 3.7+
* OpenCV – for image and video processing
* YOLOv5 – for face detection
* scikit-learn – for color clustering
* Tkinter – for the GUI
* Pandas, NumPy – for data handling
###🚀 How to Run the Code
1. Clone the Repo
   git clone https://github.com/yourusername/truetone.git
   cd truetone
2. Create Virtual Environment
   python -m venv venv
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
3. Install Dependencies
   Make sure you have all required libraries:
   pip install -r requirements.txt
   If requirements.txt isn't available, you can manually install:
   pip install opencv-python scikit-learn pandas numpy pillow
   For YOLOv5, follow installation instructions from YOLOv5 GitHub
4. Run the App
   python final_code.py
###📊 Dataset
The Final_Foundation_dataset.csv includes:
* Brand & product names
* Shade names
* Corresponding RGB values
* Pre-processed for direct color comparison

###🎯 Use Case Scenarios
* 💄 Users: Try on makeup virtually before purchasing
* 🛍️ Retailers: Embed this for personalized customer experiences
* 🧪 Researchers: Experiment with skin detection, tone mapping, and dataset expansion

###🤝 Contributing
This project was developed as part of a AI and Machine Learning class. Contributions are welcome! Please feel free to submit pull requests or open issues for improvements.

###📄 License
This project is licensed under the MIT License – see the LICENSE file for details.

###👥 Authors
* Tessa Correig
* Bernarda Andrade
* Paula Evangelista
* Niccoló Pragliola
* Sofía Serantes

###🙏 Acknowledgments
* RUBÉN SÁNCHEZ GARCÍA for guidance and support
* IE University AI and Machine Learning Course

