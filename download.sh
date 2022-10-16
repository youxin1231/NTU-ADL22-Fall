if [ ! -f ./ckpt/intent/best.pt ]; then
  wget url -O ./ckpt/intent/best.pt.zip
  unzip ./ckpt/intent/best.pt.zip
fi
if [ ! -f ./ckpt/slot/best.pt ]; then
  wget url -O ./ckpt/slot/best.pt.zip
  unzip ./ckpt/slot/best.pt.zip
fi