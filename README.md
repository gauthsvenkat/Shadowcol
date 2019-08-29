# Shadowcol

A python control interface to play games and control your PC using voice commands

Created by [@andohuman](https://twitter.com/andohuman) | [u/andohuman](https://www.reddit.com/user/andohuman)

## Table of contents

* [1. Getting Started](#1-getting-started)
* [2. Installation](#2-installation)
    * [2.1 Requirements](#21-requirements)
        * [2.1.1 Cloning the repository](#211-cloning-the-repository)
        * [2.1.2 Installing requirements](#212-installing-requirements)
        * [2.1.3 Installing PyAudio package (Windows)](#213-installing-pyaudio-package-windows)
        * [2.1.3 Installing PyAudio package (Linux)](#213-installing-pyaudio-package-linux)
* [3. Training the model](#3-training-the-model)
    * [3.1 Generate the training data](#31-generate-the-training-data)
    * [3.2 (optional) Generate the validation data](#32-optional-generate-the-validation-data)
    * [3.3 Train the model](#33-train-the-model)
* [4. (optional) Validating the trained model](#4-optional-validating-the-trained-model)
* [5. Running the program](#5-running-the-program)
    * [5.1 Providing the references](#51-providing-the-references)
    * [5.2 Executing the program](#52-executing-the-program)
* [6. Using your own voice commands](#6-using-your-own-voice-commands)
* [Possible usecases](#possible-usecases)
* [Known Issues](#known-issues)
* [Disclaimer](#disclaimer)
* [Contributing to the project](#contributing-to-the-project)

## 1. Getting started
* This program works by employing a neural network that finds out what command you say by comparing it with reference audio of the commands which you provide.
* Unlike traditional neural networks, this implementation does not require a large dataset of your voice, powerful hardware resources or a long training time. You can get this up and running typically in 10-15 minutes.
* Please note that this is a project that is still in its infancy and there are still kinks that need to be worked out. **DO NOT ATTEMPT TO PLAY ANY COMPETITIVE GAMES WITH THIS.**

Let's continue with the installation.
## 2. Installation
Note:- While this program works on both windows and linux based systems, I've personally found that the mic input from windows is much cleaner and better for the same environment and you'd usually face very little issues to get this up and running on a windows system.
### 2.1 Requirements
##### Before we begin:-
You need to have python3 64-bit and pip3 installed (and a mic, obviously).

You will need the following python libraries for this program to work.

`pyKey` - To simulate keypresses

`librosa` - To preprocess mic input

`PyAudio` - To get input from the mic

`torch` - Machine learning framework

`numpy` - For some important mathematical calculations

`auditok` - For collecting your voice commands for training

`tqdm` - To monitor progress during training
##### 2.1.1 Cloning the repository
If you have **git** installed run `git clone https://github.com/andohuman/Shadowcol.git` from your terminal.

If you don't, download the [zip file](https://github.com/andohuman/Shadowcol/archive/master.zip) and extract it.
##### 2.1.2 Installing requirements
From inside the folder, on both windows and linux, you can run the following commands from your terminal. Please note that these commands will install the **cpu only** version of pytorch.

`pip3 install -r requirements.txt`

`pip3 install torch==1.2.0+cpu torchvision==0.4.0+cpu -f https://download.pytorch.org/whl/torch_stable.html`

Note:- If you're on linux you might need to pass the `--user` flag to pip3 if you encounter any permission errors.

If you want to install the **cuda** version of pytorch you can follow the instructions on the [PyTorch website](https://pytorch.org/). (You don't need cuda or a cuda capable gpu to run this code, but if you do, you would get a nice boost in your training time)
##### 2.1.3 Installing PyAudio package (Windows)
On windows you can just run 
`pip3 install pyaudio`
##### 2.1.3 Installing PyAudio package (Linux)
On linux based systems you need to download and install PyAudio and its dependencies using your package manager.
For example,
* On Debian/Ubuntu run `sudo apt-get install python3-pyaudio`
* On Arch based systems run `sudo pacman -S python-pyaudio`
* On fedora run `sudo dnf install portaudio-devel redhat-rpm-config` then `pip3 install pyaudio --user`
## 3. Training the model
* Before you do this make sure `auditok` is in your `PATH`.

* By default the code in this repository expects 4 classes:- up, down, sil and quit. You can add or remove any command you wish. For the puposes of this tutorial we'll be training our model on these 4 classes.
### 3.1 Generate the training data
* You will need record your voice commands for the model to train on.
* You will typically need atleast 5 examples of each command (Eg:- Up, Down, etc).
* You will also explicitly need to record the ambient surrounding sound (silence) as another class.
* The recorded audio files should only contain the voice command and should typically only be a couple of milliseconds long. Please take care in ensuring that ambient noise does **NOT** precede or supercede your voice command. _**(Refer to the sample audio files in [data/examples/](data/examples/))**_

1. Make sure you're in data/train/ folder. Execute `cd data/train/` from the root directory.

2. Make sure that your mic is functioning and the surrounding sounds are favorable. You can check this by running `auditok -E` from your terminal. This command will playback your mic input and display the start and end time of your voice input in the terminal.

3. If you find that you have too much ambient noise and your mic picks that up, you can set the energy level of auditok by passing the `-e` flag. Eg:- `auditok -e 60 -E`. By default the energy value is set to 50. 

(I personally didn't have to fiddle much on my windows installation but with my linux installation I had to experiment a bit to get the right energy level because the same mic got very sensitive to the ambient noise.)

4. Once you have figured out your right energy level, you can start recording your voice commands.

For example if you're recording your voice for the 'up' command,  you can run:- `auditok -e 55 -o up_{N}.wav`. This will set the energy level to 55 and for each time you say the command, auditok will write your voice input to files named up_1.wav, up_2.wav  and so on.

**IMPORTANT: Please make sure that you set the filenames to the above mentioned format as the data loading function relies on the existence of _underscore_ in the filename.**

Play the wav audio and if you find that you have ambient noise preceding or superceding your voice command, please refer to step 2 and see sample audio [here](data/examples).

Do this respectively for 'down' and 'quit' commands. `auditok -e 55 -o down_{N}.wav` and `auditok -e 55 -o quit_{N}.wav`.

5. Now you'd need to record the ambient silence as another explicit class. You can just lower the energy levels and wait for your mic to pick up random noise. Eg:- `auditok -e 45 -o sil_{N}.wav`

You should have roughly 20 .wav files (5 examples for each command)
### 3.2 (optional) Generate the validation data
* While this step is not required for the model to work, you can generate validation data to evaluate the performance of the model after it finishes training. 
* Please note that this data will **NOT** be used for training the model.

1. Make sure you're in data/valid folder. Execute `cd data/valid/` from the root directory.

2. Exactly following the steps for generating the train data in the previous section, proceed to generate, maybe 2 examples for each of your audio commands.
### 3.3 Train the model
1. Now that you have your train data ready in data/train/  you are ready to train your model. Run `python3 train.py -e 10` from the root directory.

 This will train the model for 10 epochs which will take around 1-2 minutes depending on your configuration and save the .pth file to the model/ folder. If you're running this under cuda, this should be done in under a minute.

(I've found that usually just 5 epochs are more than enough but feel free to experiment.)
## 4. (optional) Validating the trained model
* If you've generated data to validate the model against, in step 3.2, you can run `python3 validate.py -e 10`. This will run the `validate.py` script and `-e` flag refers to using the model's 10th epoch weights. The output of this script is a bunch of lines printed showing out how similar or dissimilar two audios are, in the following format.
`<audio_file1> <audio_file2> score`

* Ideally the scores should be leaning 
    * closer to 0 if the two audios are different commands.
    * closer to 1 if the two audios are the same command. 
## 5. Running the program
* At this point you should have your model saved in model/ folder.
### 5.1 Providing the references
1. Since this model works by calculating the similarity between your voice commands, you need to provide the reference audios of your commands. You can simply choose to copy **one** file for **each** of the different commands from either `data/train/` or `data/valid/` and paste it in `references/`. Alternatively, you can also choose to record new audio files from auditok.

2. Make sure that you rename the audio files in `references/` as just `up.wav`, `down.wav`, `quit.wav`, `sil.wav`.
### 5.2 Executing the program
1. Now that you have your model and your reference audio files ready, you can begin executing the program. Run `python3 live_testing.py -e 10`. This will start the program and `-e` refers to using the mode's 10th epoch weights.

By default, the model inference will be done on the cpu. You can choose to pass the `-d cuda` flag to make inferences on your gpu. I've personally felt that inference on the cpu has much lower latency than inference on the gpu.

You can go ahead and test the program by playing the [chrome dinosaur game](http://www.trex-game.skipser.com). Saying 'quit' closes the program.
## 6. Using your own voice commands.
1. You can follow the steps mentioned in section 3 and section 4 to record your own commands, train and validate the model. You can include as many commands as you please.

2. You will then have to change a couple of lines in [live_testing.py](live_testing.py). Specifically, in lines 35-38 add your commands or remove existing ones in the `refs` dictionary the way it's specified.

The `refs` dictionary will load the reference audio files present in `references/` and preprocess them.

3. Next you'd have to change the lines from 65-70. By default, the code will have keypress conditions for only 'up', 'down' and 'quit' commands (silence will be ignored). You will have to set the keypress conditions for your own commands.

`scores` is a python list containing the respective similarity values for each of your command in the order they appear in the `refs` dictionary. For example, by default, index 0 will contain the similarity score for 'up', index 1, the scores for 'down' and so on (note that index 2 is ignored because that's silence and we don't want to do anything if it's silence). 

We find the index of the most probable command that was uttered at that instance using `np.argmax()` and simulate our keypresses accordingly. 

The `press()` function will simulate pressing the key that is passed as the argument. (Please see [pyKey](https://github.com/andohuman/pyKey#usage) usage)

Modify these lines to suite your preferences.

4. Once you made these modifications and made sure you have provided the reference audios in `references/`, you can run `python3 live_testing.py -e 10`.

You can now go ahead and launch your game.
## Possible usecases
* You can use this in addition to your normal keyboard and mouse input.
    * In FPS games you can set voice commands and map keys to switch weapons, open inventory, etc.
* You can set voice commands for certain terminal commands.
* You should also be able to use this to open/close programs.
## Known Issues
* Since the model works by comparing the similarities, sometimes external noise that sounds similar to the command might cause the program to mispress some keys. It is suggested that you run this code in a relatively quiet environment and close the program when not in use. Be careful not to rage when the program is running.

* If the program crashes or bugs out, sometimes your keyboard may act funky. This is because some key might be pressed and held down. You can fix this by just pressing all the keys you've mapped to the program.
## Disclaimer
As I've mentioned before, this project is still in its infancy and might sometimes miss your commands. There might a couple of other issues too, so I'd suggest you not use this while playing any competitive matches or doing important stuff.
## Contributing to the project
* If you have any suggestions or feature requests, feel free to contact the author.
* Play with the code and if you can make any improvements, create a pull request.
* There is another branch dedicated to get this working on [limbo](https://playdead.com/games/limbo/). Feel free to check that out too.
