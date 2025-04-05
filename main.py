from helping_functions import Dataset, CNN_model
import numpy as np

set = Dataset(data_dir="C:/Users/shrey/Desktop/ACAD/DL/inaturalist_12K/train")

model = CNN_model(
    input_size = 224,
    output_size= 10,
    num_filters = [16, 32, 64, 128, 256],
    size_filters = [3,3,3,3,3],
    paddings = [1,1,1,1,1],
    conv_strides=[1,1,1,1,1],
    pool_kernels=[2,2,2,2,2],
    dense_layer= 1000,
    activation_fn="relu",
    use_softmax=False
)
print(model)

