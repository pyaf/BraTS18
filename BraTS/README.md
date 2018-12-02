# Instructions 

`code` directory is taken from https://github.com/ieee820/BraTS2018-tumor-segmentation 

The project requires PyTorch 0.4.0 and Python 3.6.0 
other requirements are stored in `requirements.txt` file.

I've trained deepmedic model on BraTS 2018 dataset and achived a validation dice of about 0.90.
The trained model is stored in `ckpts/deepmedic_ce_50_50_fold0/` and a test scan (from validation set) to test the model performance is stored in `test/scan` to test the model and predict segmentation for this test scan, run `test.py` file 