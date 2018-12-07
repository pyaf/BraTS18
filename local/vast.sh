df -h --total
apt install htop nano unzip

mkdir data
cd data
wget https://www.cbica.upenn.edu/sbia/Spyridon.Bakas/MICCAI_BraTS/2018/MICCAI_BraTS_2018_Data_Validation.zip 
wget https://www.cbica.upenn.edu/sbia/Spyridon.Bakas/MICCAI_BraTS/2018/MICCAI_BraTS_2018_Data_Training.zip 

unzip -q -o MICCAI_BraTS_2018_Data_Training.zip -d MICCAI_BraTS_2018_Data_Training
unzip -q -o MICCAI_BraTS_2018_Data_Validation.zip -d MICCAI_BraTS_2018_Data_Validation

rm *.zip

cd ..

pip install -r requirements.txt

git clone https://github.com/thuyen/multicrop.git
cd multicrop
python setup.py install
cd ..

python split.py
python prep.py  # takes about 10 mins on 12 core CPU

python train.py --gpu 0 --cfg deepmedic_ce_all
python train.py --gpu 0 --cfg deepmedic_ce_50_50_fold0
# sftp://root@52.204.230.7:26898
#ssh -p 26898 root@52.204.230.7 -L 8080:localhost:8080

#scp -P 26898 -r code root@52.204.230.7:/home/

# notice capital P in scp and small p in ssh

