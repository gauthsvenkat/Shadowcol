if [[ $1 == "deb" ]]; then 
	sudo apt-get python3-pyaudio python3-tk -y
fi

if [[ $1 == "arch" ]]; then
	sudo pacman -S python-pyaudio tk --noconfirm
fi
pip3 install librosa matplotlib numpy

