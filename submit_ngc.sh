pip install progress;
pip install torch==1.6.0 torchvision==0.7.0
cp /workspace/imgnet.sh /tmp;
cp /dataset/*.tar /tmp;
cd /tmp && sh imgnet.sh;
python train_imagenet.py --data /tmp/ -save /workspace/edge/imagenet_tiny2 --train_batch 1024
