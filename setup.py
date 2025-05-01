from setuptools import setup, find_packages

setup(
  name='foundation_matcher',
  version='0.1.0',
  description='GUI app for matching foundation shades via skin tone analysis',
  long_description=open('README.md', encoding='utf-8').read(),
  long_description_content_type='text/markdown',
  author='Sofía Serantes',
  author_email='you@yourdomain.com',
  url='https://github.com/sofiaserantes/AI-ML--Final-Project',
  packages=find_packages(),                      # will pick up foundation_matcher/
  include_package_data=True,                     # allow CSV in package_data
  install_requires=[
    'opencv-python','numpy','Pillow','pandas',
    'scikit-learn','mediapipe'
  ],
  entry_points={
    'console_scripts': [
      'foundation-match=foundation_matcher.app:main'
    ]
  },
  python_requires='>=3.6',
)
