echo 'Downloading ImageNet...'
cp -r /mnt/home/gkrawezik/ceph/AI_DATASETS/ImageNet/2012/imagenet /tmp/imagenet/
pushd /tmp/imagenet/
mkdir val train

echo 'Unzipping training folder...'
cd train/
unzip -qq ../train.zip 

echo 'Unzipping validation folder...'
cd ../val
unzip -qq ../val.zip
popd