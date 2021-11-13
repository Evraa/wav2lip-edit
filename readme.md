## Running on ec2

## 1. Setup
* Create an environment with cuda v 11.4+ and cuDNN v 8+, better with anaconda, recent versions are needed for TF.
```shell
    $ conda create -n deepfacelab -c main python=3.8 cudnn=8.6 cudatoolkit=11.4 -c conda-forge
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

> To download any of them (eg. wav2lip_gan):

```shell
    $ wget "https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/EdjI7bZlgApMqsVoEUUXpLsBxqXbn5z8VTmoxp55YNDcIA?e=n9ljGW" -O "Wav2Lip/content/wav2lip_gan.pth"
```

## 3. Place data

> You need one .mp4 video and one audio to start with. Make sure to type in the appropriate path for them.

#### The result is saved (by default) in results/result_voice.mp4. You can specify it as an argument, similar to several other available options. The audio source can be any file supported by FFMPEG containing audio data: *.wav, *.mp3 or even a video file, from which the code will automatically extract the audio.


## 4. Run inference
```shell
    $ python inference.py --checkpoint_path checkpoints/wav2lip_gan.pth --face "../sample_data/input_vid.mp4" --audio "../sample_data/input_audio.wav"
```


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
* Runnin on cpu (worked nut very slow)
    - Takes up to one hour to predect faces on a 120 seconds video with 25 fps.
* CPU with fewer fps (5 fps)
    - Takes up to 15 mins. Results are ..


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