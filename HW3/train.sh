if [ ! -d data ]; then
    gdown https://drive.google.com/uc?id=186ejZVADY16RBfVjzcMcz9bal9L3inXC
    unzip data.zip && rm data.zip
fi