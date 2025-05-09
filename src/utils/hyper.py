
BATCHSIZES = [32, 64, 128]
LRS = [0.01, 0.001, 0.0001]
EPOCHS= [i for i in range(10, 26, 2)]

MULT_LRS = [0.001, 0.005, 0.0001, 0.0005, 0.00001, 0.00005]
MULT_PATIENCES = [3, 5, 7, 10, 15, 20]

BIN_BATCH = 64
BIN_LR = 0.0001
BIN_EPOCH = 24
BIN_CUTOFF = 0.537

MULT_BATCH = 64
MULT_LR = 0.0001
MULT_PATIENCE = 15

BASE_BATCHSIZE = 64
BASE_LR = 0.001
BIN_BASE_EPOCH = 15
MULT_BASE_PATIENCE = 5

SEED = 179180