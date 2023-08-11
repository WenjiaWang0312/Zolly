<div align="center">

<h1>Zolly: Zoom Focal Length Correctly for Perspective-Distorted Human Mesh Reconstruction </h1>
</div>


![teaser](assets/teaser.png)
The first work aims to solve 3D Human Mesh Reconstruction task in **perspective-distorted images**. 



# 🗓️ News:

🎆 2023.Aug.7, the dataset link is released. The training code is coming soon.

🎆 2023.Jul.14, Zolly is accepted to ICCV2023, codes and data will come soon.

🎆 2023.Mar.27, [arxiv link](https://arxiv.org/abs/2303.13796) is released.


# 🚀 Run the code
## 🌏 Environments
- You should install [`MMHuman3D`](https://github.com/open-mmlab/mmhuman3d/blob/main/docs/install.md) firstly.


You can install pytorch3d from file if you find any difficulty. 

```
wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch3d/linux-64/pytorch3d-0.7.4-py310_cu118_pyt200.tar.bz2

pip install fvcore

pip install iopath

conda install --use-local pytorch3d-0.7.4-py{38}_cu{117}_pyt{1131}.tar.bz2
```

- install this repo
```
cd Zolly
pip install -e .
```

## 💾 Dataset Preparation
- 💿 Preprocessed npzs, all has ground-truth focal length, translation and smpl parameters. It is easy to load the annotations with `np.load(path)`. We will release training code as soon as possible.
    - [HuMMan](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/wwj2022_connect_hku_hk/EuXCqmz3v6dFslQGwv9eRyUBywmMDqoUGUuoxOVp1UeDzA) (train, test_p1, test_p2, test_p3
    - [SPEC-MTP](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/wwj2022_connect_hku_hk/Er8fPdOE5mJNvX0zswUal8IBTq2rYk7lhiZFeCuNFFh-hw) (test_p1, test_p2, test_p3)
    - [PDHuman](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/wwj2022_connect_hku_hk/Eln52WC8rSJLk8A6hiC9msUBMlTbB4b65OdyXIX4YoBqsQ) (train, test_p1, test_p2, test_p3, test_p4, test_p5)
    - [3DPW](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/wwj2022_connect_hku_hk/Egf4YuLUKbtOjK6lP3G2X1UB2vEptMR5cJpE_4-1Zq6Qyg) (train(has optimized neutral betas), test_p1, test_p2, test_p3)

- 🌁 Images.
    - [HuMMan](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/wwj2022_connect_hku_hk/EhQf5Z37_Y5EoeiEJRL3kEEBM9bjlPo5edJ4djMb8jbatw)
    - [SPEC-MTP](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/wwj2022_connect_hku_hk/EqBRcsqLt0BHjeE254JhFHIBtsfpqDofFaT3QQf5-QWtkQ)
    - [PDHuman](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/wwj2022_connect_hku_hk/EjGl9svxV_xHoC0hHlHVpcMB7IZwYbyiFVbS8iRP9cVsIg)

## 🚅 Train

## 📺 Test & Demo

## 💻Add Your Algorithm


# 🎓 Citation

If you find this project useful in your research, please consider cite:

```
@article{wang2023zolly,
  title={Zolly: Zoom Focal Length Correctly for Perspective-Distorted Human Mesh Reconstruction},
  author={Wang, Wenjia and Ge, Yongtao and Mei, Haiyi and Cai, Zhongang and Sun, Qingping and Wang, Yanjun and Shen, Chunhua and Yang, Lei and Komura, Taku},
  journal={arXiv preprint arXiv:2303.13796},
  year={2023}
}
```
# 📧 Contact

Feel free to contact me for other questions or cooperation: wwj2022@connect.hku.hk
