<h1 align="center">
<p><a href='https://arxiv.org/pdf/2206.09959v1.pdf'>GCViT: Global Context Vision Transformer</a></p>
</h1>
<div align=center><img src="https://raw.githubusercontent.com/awsaf49/gcvit-tf/main/image/lvg-arch.PNG" width=800></div>
<p align="center">
<a href="https://github.com/TensorSpeech/TensorFlowASR/blob/main/LICENSE">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg">
</a>
<img alt="python" src="https://img.shields.io/badge/python-%3E%3D3.6-blue?logo=python">
<img alt="tensorflow" src="https://img.shields.io/badge/tensorflow-%3E%3D2.5.1-orange?logo=tensorflow">
<h2 align="center">
<p>Tensorflow 2.0 Implementation of GCViT</p>
</h2>
</p>
<p align="center">
This library implements <b>GCViT</b> using Tensorflow 2.0 specifally in <code>tf.keras.Model</code> manner to get PyTorch flavor.
</p>


## Model
* Architecture:

<img src="https://raw.githubusercontent.com/awsaf49/gcvit-tf/main/image/arch.PNG">

* Local Vs Global Attention:

<img src="https://raw.githubusercontent.com/awsaf49/gcvit-tf/main/image/lvg-msa.PNG">

## Result
> The reported result in the paper is shown in the figure. But due to issues in the **codebase** actual result differs from the reported result.

<img src="https://raw.githubusercontent.com/awsaf49/gcvit-tf/main/image/result.PNG" width=900>

## Installation
```bash
pip install -U gcvit
# or
# pip install -U git+https://github.com/awsaf49/gcvit-tf
```

## Usage
Load model using following codes,
```py
from gcvit import GCViTTiny
model = GCViTTiny(pretrain=True)
```
Simple code to check model's prediction,
```py
from skimage.data import chelsea
img = tf.keras.applications.imagenet_utils.preprocess_input(chelsea(), mode='torch') # Chelsea the cat
img = tf.image.resize(img, (224, 224))[None,] # resize & create batch
pred = model(img).numpy()
print(tf.keras.applications.imagenet_utils.decode_predictions(pred)[0])
```
Prediction:
```py
[('n02124075', 'Egyptian_cat', 0.9194835),
('n02123045', 'tabby', 0.009686623), 
('n02123159', 'tiger_cat', 0.0061576385),
('n02127052', 'lynx', 0.0011503297), 
('n02883205', 'bow_tie', 0.00042479983)]
```

## To Do
- [ ] Working training example.
- [ ] Gradio Demo.
- [x] Build model with `tf.keras.Model`.
- [x] Port weights from official repo.

## Acknowledgement
* [GCVit](https://github.com/NVlabs/GCVit) (Official)
* [Swin-Transformer-TF](https://github.com/rishigami/Swin-Transformer-TF)
* [tfgcvit](https://github.com/shkarupa-alex/tfgcvit/tree/develop/tfgcvit)
* [keras_cv_attention_models](https://github.com/leondgarse/keras_cv_attention_model)


## Citation
```bibtex
@article{hatamizadeh2022global,
  title={Global Context Vision Transformers},
  author={Hatamizadeh, Ali and Yin, Hongxu and Kautz, Jan and Molchanov, Pavlo},
  journal={arXiv preprint arXiv:2206.09959},
  year={2022}
}
```