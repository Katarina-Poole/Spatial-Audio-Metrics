# Spatial-Audio-Metrics
Spatial Audio Metrics (SAM) is a toolbox to analyse spatial audio and spatial audio perceptual experiments

The demo folder contains scripts showing you how to use the package


# To test the package works locally
1. cd to repo
2. pip install -e .    (the e means edit so you can edit the source package as is without having to reinstall)
3. Test out functions (package is called)

# To build the package and upload to pip (when ready to release 0.0.1). Make sure the command prompt is in the package directory
1. python -m build
2. Upload to test.pypi.org to check it first (make sure you have an account). Need to pip install twine to do this
3. python -m twine upload --repository testpypi dist/*
4. pip install -i https://test.pypi.org/simple/ spatialaudiometrics
5. Then check that runs and installs. 
6. When happy that everything is working correctly then run:
python -m twine upload --repository pypi dist/*
7. Then should be able to see python package uploaded 

pip install spatialaudiometrics

# For documentation
pip install sphinx
pip install sphinx-rtd-theme

To build:
sphinx-build -M html C:\GitHubRepos\Spatial-Audio-Metrics\spatialaudiometrics C:\GitHubRepos\Spatial-Audio-Metrics\docs

or can run 
sphinx-apidoc -o docs spatialaudiometrics 

then cd to docs and run
make html


For now can just run `make html` in the docs folder

# License
Copyright (C) 2024  Katarina C. Poole

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
