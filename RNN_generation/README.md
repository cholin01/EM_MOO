# RNNMGM

1. data preprocess

   you can use the dataset int data/

   or using  utils/augmentation.py for data preprocess

2. train

   generation model(launcher_of_clm.py)

```
1. pretain:
	train_clm(data_path='data/Ds_9.csv', smi_idx=0, model_name='pt', epochs=30, fq_saving=5)
2. fine-tune:
	train_clm(data_path='data/Dm.csv', smi_idx=0, model_name='tl', epochs=30, fq_saving=5)

```


3 . generation 

```
run the generate() or valid_generate() in launcher_of_clm.py

```


