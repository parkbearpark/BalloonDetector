from setuptools import setup, find_packages

setup(
    name='BalloonDetector',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'opencv-python',
        'tensorflow-macos',
        'pillow',
        'matplotlib'
    ],
    include_package_data=True,
    author='Ito Yuma',
    author_email='yumai6205@gmail.com',
    description='A package to detect balloons in pages of Japanese manga.',
    url='https://github.com/parkbearpark/BalloonDetector.git',
)
