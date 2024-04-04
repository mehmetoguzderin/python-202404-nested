conda env create --prefix ./conda -f environment.yml
conda activate ./conda
pip install -e .
bash scripts/download_models.sh
export AM_I_DOCKER=False
export BUILD_WITH_CUDA=False
cd Grounded-Segment-Anything
python -m pip install -e segment_anything
python -m pip install -e GroundingDINO
cd ..
