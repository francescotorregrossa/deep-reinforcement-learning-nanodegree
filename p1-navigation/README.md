```bash
cd p1-navigation

conda create --name p1_navigation python=3.6
conda activate p1_navigation
unzip setup.zip
cd setup
pip install .
cd ..

python main.py

conda deactivate

conda remove --name p1_navigation --all
```