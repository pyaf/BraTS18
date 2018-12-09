## overall survival

### install cmake version 3.10 +

wget https://github.com/Kitware/CMake/releases/download/v3.13.1/cmake-3.13.1.tar.gz
tar -xvf cmake-3***
cd cmake**
./configure && make && sudo make install

cmake --version


### install ants
mkdir ~/bin/ants && cd ~/bin/ants
cmake /media/ags/CODE/GE/code/BraTS/BraTS2018-survival-prediction/code/ANTs
make -j 2

export ANTSPATH=${HOME}/bin/ants/bin/
export PATH=${ANTSPATH}:$PATH

For Ubuntu/debian: http://neuro.debian.net/install_pkg.html?p=fsl-complete
else: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation/Linux

 sudo apt-get install libopenblas-base