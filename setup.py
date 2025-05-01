import os
import sys
from setuptools import setup, find_packages

# Read the README for the long description

def readme():
    here = os.path.abspath(os.path.dirname(__file__))
    readme_path = os.path.join(here, 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, encoding='utf-8') as f:
            return f.read()
    return ''

# Package version
VERSION = '0.1.0'

# Dependencies required to run the application
install_requires = [
    'opencv-python',      # for cv2
    'numpy',              # for numeric operations
    'Pillow',             # for ImageTk
    'pandas',             # for dataset handling
    'scikit-learn',       # for KMeans clustering
    'mediapipe'           # for face detection
]

setup(
    name='foundation_matcher',
    version=VERSION,
    description='GUI app for matching foundation shades via skin tone analysis',
    long_description=readme(),
    long_description_content_type='text/markdown',
    author='Your Name',
    author_email='you@example.com',
    url='https://github.com/yourusername/foundation_matcher',
    py_modules=['final_version_code'],
    packages=find_packages(),
    install_requires=install_requires,
    entry_points={
        'console_scripts': [
            'foundation-match=final_version_code:main'
        ],
    },
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
