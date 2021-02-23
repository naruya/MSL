# MSL
Synergetic Model and Skill Learning for Unsupervised Agent Pretraining

## Preparation

docker & [pyenv](https://github.com/pyenv/pyenv-installer) is highly recommended

### 1. install [mujoco-py<1.50.2,>=1.50.1](https://github.com/openai/mujoco-py/tree/1.50.1.0)
```
# place mjpro150 and mjkey.txt into ~/.mujoco/
$ apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    curl \
    git \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libosmesa6-dev \
    python3-pip \
    python3-numpy \
    python3-scipy \
    unzip \
    vim \
    wget \
    xpra \
    xserver-xorg-dev
$ curl -o /usr/local/bin/patchelf https://s3-us-west-2.amazonaws.com/openai-sci-artifacts/manual-builds/patchelf_0.9_amd64.elf \
    && chmod +x /usr/local/bin/patchelf
$ echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mjpro150/bin' >> /root/.zshrc
$ source ~/.zshrc
$ pyenv install 3.5.10
$ pyenv virtualenv 3.5.10 diayn
$ pyenv global diayn
$ pip install -U pip
$ git clone https://github.com/openai/mujoco-py.git
$ pip install -r mujoco-py/requirements.txt
$ pip install -r mujoco-py/requirements.dev.txt
$ pip install -U 'mujoco-py<1.50.2,>=1.50.1'
```
### 2. install [gym\[all\]==0.10.5](https://github.com/openai/gym/tree/v0.10.5)
```
$ pip install 'gym[all]'
```

### 3. install [rlkit](https://github.com/naruya/DIAYN/tree/21916bb66cdef6622c512cc3d33d0f93a0af55e4)
```
$ git clone --recursive https://github.com/naruya/MSL.git
$ cd MSL/diayn
$ apt-get install swig
$ pip install -r requirements.txt
$ pip install -e .
```

### 4. Test (mujoco, mujoco-py, gym, rlkit)
```
# train
$ python examples/diayn.py HalfCheetah-v2
# eval
$ python scripts/run_policy_diayn.py HalfCheetah-v2 data/path/to/params.pkl
```
