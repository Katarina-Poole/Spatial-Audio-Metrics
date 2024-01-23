# Spatial-Audio-Metrics
Spatial Audio Metrics (SAM) is a toolbox to analyse spatial audio and spatial audio perceptual experiments

The demo folder contains scripts showing you how to use the package


# To test the package works locally
1. cd to repo
2. pip install .
3. Test out functions (package is called)

# To build the package and upload to pip (when ready to release 0.0.1). Make sure the command prommpt is in the package directory
1. python -m build
2. Upload to test.pypi.org to check it first (make sure you have an account). Need to pip install twine to do this
3. python -m twine upload --repository testpypi dist/*
4. pip install -i https://test.pypi.org/simple/ spatialaudiometrics
5. Then check that runs and installs. 
6. When happy that everything is working correctly then run:
python-m twine upload --repository pypi dist/*
7. Then should be able to see python package uploaded 

pip install spatialaudiometrics