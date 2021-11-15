## Running on ec2

## 1. Setup
* Create an environment with cuda v 11.4+ and cuDNN v 8+, better with anaconda, recent versions are needed for TF.
```shell
    $ conda create -n wav2lip -c main python=3.8 cudnn=8.0 cudatoolkit=11.0 -c conda-forge
```
* Clone the repo
```shell
    $ git clone https://github.com/Rudrabha/Wav2Lip.git
```
* Install ffmpeg:
```shell
    $ sudo apt-get install ffmpeg   
```
* Install requirements:
```shell
    $ pip install -r requirements_mod.txt
```
> NOTE: these reqs are modified. To be compatible with EC2

> Also, make sure that opencv-python is installed properlly.

* Install torch
```shell
    $ pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
```
## 2. Download models

* Download `s3fd` model for face detection **(86 MB)**
```shell
    $ wget "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth" -O "Wav2Lip/face_detection/detection/sfd/s3fd.pth"
```
> NOTE: It's being placed on a specific path, with *s3fd.pth* name, so make sure to run this command on the directory before *Wav2Lip*.


* Inference models, they provide four different models to work with:

Getting the weights
----------
| Model  | Description |  Link to the model | 
| :-------------: | :---------------: | :---------------: |
| Wav2Lip  | Highly accurate lip-sync | [wav2lip.pth **(416 MB)**](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/Eb3LEzbfuKlJiR600lQWRxgBIY27JZg80f7V9jtMfbNDaQ?e=TBFBVW)  |
| Wav2Lip + GAN  | Slightly inferior lip-sync, but better visual quality | [wav2lip_gan.pth **(416 MB)**](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/EdjI7bZlgApMqsVoEUUXpLsBxqXbn5z8VTmoxp55YNDcIA?e=n9ljGW) |
| Expert Discriminator  | Weights of the expert discriminator | [lipsync_expert.pth **(188 MB)**](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/EQRvmiZg-HRAjvI6zqN9eTEBP74KefynCwPWVmF57l-AYA?e=ZRPHKP) |
| Visual Quality Discriminator  | Weights of the visual disc trained in a GAN setup | [visual_quality_disc.pth **(162 MB)**](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/EQVqH88dTm1HjlK11eNba5gBbn15WMS0B0EZbDBttqrqkg?e=ic0ljo) |

> To download any of them (eg. wav2lip_gan): You can manually use the link, and make sure to place it at ./checkpoints/

> If you want to download it using a script. I provided a script `dl.py` very easy and simple to use and read.
```shell
    $ pip install gdrive
    $ python3 dl.py
```

## 3. Place data

> You need one .mp4 video and one audio to start with. Make sure to type in the appropriate path for them.

#### The result is saved (by default) in results/result_voice.mp4. You can specify it as an argument, similar to several other available options. The audio source can be any file supported by FFMPEG containing audio data: *.wav, *.mp3 or even a video file, from which the code will automatically extract the audio.


## 4. Run inference
```shell
    $ python inference.py --checkpoint_path checkpoints/wav2lip_gan.pth --face "../sample_data/input_vid.mp4" --audio "../sample_data/input_audio.wav"
```

### Advanced
> You may use parameters provided by the authors such as --box, --nosmoot ..etc. We'll discuss some of them in the following subsection:

* *--box:* To exclude the `s3fd` face detector model and manually locate the face within the video/image. Works better in case of images to save time. But authors forgot to implement it for images, so I did it in the modified version in this repo.
* *--crop:* To crop the image and work on a smaller scale.
* *--nosmooth:* Proved to show better results. Recommended.
* *--fps:* Doesn't provide its intended job, so I implemented it within the code.
* *--resize_factor:* Since modek is trained on small scale images, and produce at most 512*512 videos, so scalling down is required for HD images/videos. This paramtere is a number larger than one to divide width and length on.

# Issues that come along the way ya zmeely

## Image too big!
```python3
    RuntimeError: CUDA error: an illegal memory access was encountered
    
    Then..

    RuntimeError: Image too big to run face detection on GPU. Please use the --resize_factor argument
```
### Sol:
* --resize_factor: larger than 1 (eg. 2, 10, 100):
    - Doesn't always work .. needs a lot of trial & error based on your video aspects.
    - If image is still big, same error will occur, if image is very tiny .. another error will occur x')
* Using a bounded box (semi-success)
    - Needed to be tuned exactly on the face, otherwise it will produce awful results.
* Runnin on cpu (worked nut very slow - 17 s/it - it=1 img (no batching in cpu))
    - Takes up to one hour to predect faces on a 120 seconds video with 25 fps.
* CPU with fewer fps (5 fps)
    - Takes up to 15 mins. Results are ..
    - Problem is that the argument `fps` is kinda useless, to adjust the videos fps .. you need to modify it manually. This version
        of the code is modified, it will ask you for the fps while running.
* Run on smaller images (works - bad results/pixled)
    - I guess the limit is 512*512!

## Can't find temp/temp.wav!
```python3
    RuntimeError: Error opening 'temp/temp.wav': System error.

    Then..

    FileNotFoundError: [Errno 2] No such file or directory: 'temp/temp.wav'
```
### Sol:
* reinstalling ffmpeg (failed)
    ```shell
        $ sudo apt-get install ffmpeg  
    ```
* executing within `Wav2lip` directory (success)
    The problem is that I ran the `inference.py` file from another directory. It was a relative path error, not a package one.


## Limits
> *"All results are currently limited to (utmost) 480p resolution and will be cropped to max. 20s to minimize compute latency"* ~said the author.

### Sol:
* Ask the authors for HD model. They did not release it.

## Unnecessary termination!
    If One frame doesn't contain an image, the whole process is discarded.

### Sol:
* Try modifying the code and list index length if you want to.
* Use an image instead of a video. (Works)

# Results:


---
---

# Upgrades

## Running PaddleGAN

PaddleGAN is a very comprehensive api consists of various generative models, collecting projects from different sources.
Wav2lip is one of these projects, notice that they are implementing the execution of the same wav2lip [link](https://github.com/Rudrabha/Wav2Lip.git), in addition to the feature of face enhancement, that paddlegan produce [here](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/ppgan/faceutils/face_enhancement/face_enhance.py).

> So in a nutshil, PaddleGAN Wav2lip = Wav2lip + face enhancement

### How to run
* Follow the same instructions provided above.
* Install PaddlePaddle [follow here for more](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/docs/zh_CN/install.md)
```shell
    # CUDA10.1
    $ python -m pip install paddlepaddle-gpu==2.1.0.post101 -f https://mirror.baidu.com/pypi/simple
    # CPU
    $ python -m pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple
```
* Execute
```shell
    $ cd applications

    $ python tools/wav2lip.py \
        --face ../docs/imgs/mona7s.mp4 \
        --audio ../docs/imgs/guangquan.m4a \
        --outfile pp_guangquan_mona7s.mp4 \
        --face_enhancement
```
> mona7s.mp4 and guangquan.m4a exist already in the repo.
> face_enhancement will automatically download the model produced by paddlegan **(270+ MB)**

## Issues On PaddleGAN wav2lip

## numpy package
```python3 
    RuntimeError: module compiled against API version 0xe but this version of numpy is 0xd
```
### Sol
* Upgrade package (solved)
```shell
    $ pip install numpy --upgrade
```

## Issues with cuDNN

### Sol:
1. Run on google colab instead (provided with the code).
2. Run on cpu
> Pass the parameter `--cpu` to the above and will work but slow (takes up to 40 mins for 5 sec videos).
3. Use docker image (didn't try it)

## Some helpful commands that might work for you

* To install cuDNN on conda and check it:
```shell
    conda install -c anaconda cudnn
    conda list cudnn
```

# Results

## Without face enhancement:



https://user-images.githubusercontent.com/41262384/141814218-36dc8fc7-d2d5-4a03-85de-ec12114d4677.mp4


## With face enhancement:

