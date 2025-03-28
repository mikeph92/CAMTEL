# conda create -n rapids-env -c rapidsai -c nvidia -c conda-forge \
#     rapids=23.10 python=3.9 cudatoolkit=11.8

conda activate rapids-env

pip install pandas scikit-image matplotlib seaborn joblib

python clustering.py

