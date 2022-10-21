# Attention Style Transfer

This repository contains a **tensorflow-based** implementation of the the arbitrary image/video style transfer method, based on an improved attention system, proposed by *Liu et al.* in the paper [AdaAttN: Revisit Attention Mechanism in Arbitrary Neural Style Transfer](https://arxiv.org/abs/2108.03647).

## Results

The following results are examples generated using a model trainet on **COCO** and **Wikiart** datasets for 36k steps. Its weights can be found on the *weights* folder and they are loaded by default when running the provided style transfer scripts.

## Image style transfer

![Image style transfer results](https://github.com/gaetano-signorelli/Video-Style-Transfer/blob/main/results/images/Examples.png)

As it can be noticed, the network is able to produce high quality style transfers, working also on low-level features. Moreover, it applies the style into the content in a way that does not change the content itself (for example by adding style's features by force), but integrating their features where it is more appropriate, thanks to the **attention** mechanism. Scaling up the resolutions leads to great benefits to the level of details: these results have been obtained with a *1024x1024* resolution.

## Video style transfer

<picture>
  <img alt="Style" src="https://github.com/gaetano-signorelli/Video-Style-Transfer/blob/main/results/videos/style.jpg", width=1280, height=720>
</picture>

![Original video content](https://github.com/gaetano-signorelli/Video-Style-Transfer/blob/main/results/videos/content.gif)

![Video style transfer results](https://github.com/gaetano-signorelli/Video-Style-Transfer/blob/main/results/videos/result.gif)

The style transfer is also quite good when applied to videos, and, also in this case, scaling up the resolutions brings great improvements with itself, with a noticeable reduced flickering effect, even without the use of a specific regularization term during the training. This example has been generated using a *1080x720* resolution and an interpolation level of *0.75*.

## Run image style transfer

To create a stylized content of a given image, run the `image_style_transfering.py` script:

`python image_style_transfering.py "content_path" "style_path" "result_path"`

Optional arguments are:
- `--w` : set width output resolution (integer value, it should be a multiple of 16, default=512)
- `--h` : set height output resolution (integer value, it should be a multiple of 16, default=512)
- `--mix` : set the interpolation level between the content image and the stylized content (float value in range [0.0, 1.0], default=1.0)
- `--cpu` : this forces the model to run on CPU; which will take much more time but it could be necessary in case of too high resolutions that cannot be handled by the GPU

## Run video style transfer

To genearate the stylized version of a given video, run the `video_style_transfering.py` script:

`python video_style_transfering.py "content_path" "style_path" "result_path"`

The parameters are the same used for the image style transfer (even the optional ones), with the only difference being the files' format.

## Train the model

### Download datasets

To be able to launch a training session, a couple of training datasets must be downloaded and placed inside the *data* folder. If their names do not match the ones on the *src/architecture/config.py* file, this one should be edited accordingly. Default names are *"coco dataset"* and *"wikiart dataset"*, since the model has been trained on them, for content and style respectively.

Particularly, they can be easily downloaded, among other online resources (including the official ones), from [Kaggle](https://www.kaggle.com/datasets/shaorrran/coco-wikiart-nst-dataset-512-100000); where a selected set of 50k images (for both datasets) has been gathered and already resized into a *512x512* resolution, saving space and time.

### Run the training

In order to train the model from scratch (or starting from the last training step), it is enough to run the command:

`python train.py`

Beforehand, training parameters can be adjusted by accessing the file *src/architecture/config.py*
