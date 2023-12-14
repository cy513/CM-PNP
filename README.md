## Environment
    python 3.8
    pytorch 2.0.0


## Run the experiment

Train the model.

    python trainCombModel.py --batch_size=1024 --comb_model=1 --dataset=DATA_NAME --gpu=1 --hidden_dim=200 --lr=0.001
    (DATA_NAME=ICEWS05, ICEWS14 or ICEWS18)

Test the model.

    python testCombModel.py --batch_size=1024 --comb_model=1 --dataset=DATA_NAME --gpu=1
    (DATA_NAME=ICEWS05, ICEWS14 or ICEWS18)


