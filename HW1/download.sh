if [ ! -f ./ckpt/intent/best.pt ]; then
  mkdir -p ./ckpt/intent
  wget https://www.dropbox.com/s/7ktvy8tygfsq18m/best_intent.pt.zip?dl=1 -O ./ckpt/intent/best.pt.zip
  unzip ./ckpt/intent/best.pt.zip -d ./ckpt/intent && rm ./ckpt/intent/best.pt.zip
  mv ./ckpt/intent/0.92177.pt ./ckpt/intent/best.pt
fi
if [ ! -f ./ckpt/slot/best.pt ]; then
  mkdir -p ./ckpt/slot
  wget https://www.dropbox.com/s/l3id1dv2i8y9824/best_slot.pt.zip?dl=0 -O ./ckpt/slot/best.pt.zip
  unzip ./ckpt/slot/best.pt.zip -d ./ckpt/slot && rm ./ckpt/slot/best.pt.zip
  mv ./ckpt/slot/0.78337.pt ./ckpt/slot/best.pt
fi