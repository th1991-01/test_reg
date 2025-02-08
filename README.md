install
```
conda create -n hishinuma python==3.12
pip install -U "jax[cuda12]"
pip install flax

pip install scikit-learn matplotlib

pip install gymnasium==0.29.1
pip install gymnasium[mujoco]
```

test env
```
$ python -c "import jax;print(jax.devices())"
##  [CudaDevice(id=0)] が表示されればOK
```

test reg
```
python get_dataset_gym.py
python reg_jax.py
```
