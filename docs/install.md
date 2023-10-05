# Getting Start

``` shell
conda create -n urbangiraffe4render python==3.8
conda activate urbangiraffe4render
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch # Install other pytorch
pip install opencv-python pyyaml numpy imageio matplotlib easydict einops scipy tqdm termcolor tensorboardX trimesh scikit-learn # Install other pkgs

export CUDA_VERSION=$(nvcc --version| grep -Po "(\d+\.)+\d+" | head -1)
pip install dist/voxrender-0.0.0-cp38-cp38-linux_x86_64.whl # Install ray-voxel intersection kernel
# if your CUDA_VERSION is not 11.1, then use following command
cd tools/voxlib
python setup.py build
python setup.py install

conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
conda install pytorch3d -c pytorch3d # Install pytorch3d
```