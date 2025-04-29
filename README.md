# TrueTone: Precision Matching for your Unique Skin
TrueTone is a computer vision-powered tool that helps users find their perfect foundation shade by analyzing skin tone through their webcam. It uses MediaPipe for real-time face detection and machine learning-based color clustering to match your skin to a curated dataset of foundation products from top beauty retailers.

## Why TrueTone?
Finding the right foundation online is a frustrating process filled with trial, error, and wasted products. 
TrueTone solves this problem by:
* Capturing your skin tone via webcam
* Analyzing your skin's undertones using color clustering
* Matching it with the closest foundation shades from top beauty retailers
* Providing a smooth and user-friendly GUI experience

## Features
### Computer Vision & ML
* MediaPipe for real-time face detection
* Skin tone sampling and KMeans clustering
* RGB color distance-based shade matching
* Top 3 closest shades recommended instantly
  
### Real-Time GUI
* Live webcam capture with face detection
* Interactive window showing shade matches
* Clear layout with foundation product details

### Foundation Dataset
* Curated CSV dataset of foundation shades
* Contains product names, brands, and RGB color values
* Built for easy scalability across retailers
###  Project Structure
```
├── final_code.py               # Main application logic
├── Final_Foundation_dataset.csv # Curated foundation shade dataset
├── README.md                   # Documentation
└── requirements.txt            # Packages needed
```
### Tech Stack
*numpy==1.25.0
* opencv-python==4.9.0.66
* mediapipe==0.10.0
* pillow==10.0.0
* scikit-learn==1.3.0
* pandas==2.1.0
  
### How to Run the Code
1. Clone the Repo
   ```
   git clone https://github.com/yourusername/truetone.git
   cd truetone
   ```
3. Create Virtual Environment
   ```
   python -m venv venv
   python -m venv venv
   Windows
   venv\Scripts\activate
   macOS/Linux
   source venv/bin/activate
   ```
5. Install Dependencies
   Make sure you have all required libraries:
   ```
   pip install -r requirements.txt
   ```
   If requirements.txt isn't available, you can manually install:
   ```
   pip install opencv-python scikit-learn pandas numpy pillow
   For YOLOv5, follow installation instructions from YOLOv5 GitHub
   ```
7. Run the App
   ```
   python final_code.py
   ```

### GUI Walkthrough
* Live feed from your webcam
* Face detection via MediaPipe
* Skin color extracted from cheek regions
* Best-match foundation shades displayed with:
* Brand & product name
  * RGB preview swatch
    
### Dataset
The Final_Foundation_dataset.csv includes:
* Brand & product names
* Shade names
* Corresponding RGB values
* Used to match user skin color via color distance

### Use Case Scenarios
* Users: Try on makeup virtually before purchasing
* Retailers: Embed this for personalized customer experiences
* Researchers: Experiment with skin detection, tone mapping, and dataset expansion

### Contributing
This project was developed as part of a AI and Machine Learning class. Contributions are welcome! Please feel free to submit pull requests or open issues for improvements.

### License
This project is licensed under the MIT License – see the LICENSE file for details.

### Authors
* Tessa Correig
* Bernarda Andrade
* Paula Evangelista
* Niccoló Pragliola
* Sofía Serantes

### Acknowledgments
* RUBÉN SÁNCHEZ GARCÍA for guidance and support
* IE University AI and Machine Learning Course

