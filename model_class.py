import torch 

class CNN_model(torch.nn.Module):
    def __init__(self, input_size, output_size, num_filters, size_filters, pool_kernels, paddings, conv_strides, dense_layer, activation_fn, use_softmax=False):
        super().__init__()

        self.conv_blocks = torch.nn.ModuleList()

        def identify_aactivation(function_name):
            activation = {
                'relu': torch.nn.ReLU(),
                'sigmoid': torch.nn.Sigmoid(),
                'tanh': torch.nn.Tanh(),
                'selu': torch.nn.SELU(),
                'gelu': torch.nn.GELU(),
                'mish': torch.nn.Mish(),
                'leakyrelu': torch.nn.LeakyReLU()
            }
            return activation.get(function_name.lower(), torch.nn.ReLU())
        
        def make_tuple(a):
            if isinstance(a, tuple):
                return a
            else:
                return (a,a)

        for i in range(len(num_filters)):
            if i == 0:
                block = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=3, 
                        out_channels=make_tuple(num_filters[i]), 
                        kernel_size=make_tuple(size_filters[i]), 
                        padding=make_tuple(paddings[i]), 
                        stride=make_tuple(conv_strides[i])
                    ),
                    identify_aactivation(activation_fn),
                    torch.nn.MaxPool2d(kernel_size=make_tuple(pool_kernels[i]), stride=make_tuple(pool_kernels[i]))
                )
            else:
                block = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=make_tuple(num_filters[i-1]), 
                        out_channels=make_tuple(num_filters[i]), 
                        kernel_size=make_tuple(size_filters[i]), 
                        padding=make_tuple(paddings[i]), 
                        stride=make_tuple(conv_strides[i])
                    ),
                    identify_aactivation(activation_fn),
                    torch.nn.MaxPool2d(kernel_size=make_tuple(pool_kernels[i]), stride=make_tuple(pool_kernels[i]))
                )
            self.conv_blocks.append(block)

        h, w = make_tuple(input_size)
        for i in range(len(num_filters)):
            f_h, f_w = make_tuple(size_filters[i])
            s_h, s_w = make_tuple(conv_strides[i])
            p_h, p_w = make_tuple(paddings[i])

            h = ((h - f_h + 2*p_h)//s_h) + 1
            w = ((w - f_w + 2*p_w)//s_w) + 1

            pp_h, pp_w = make_tuple(pool_kernels[i])
            ps_h, ps_w = make_tuple(pool_kernels[i])

            h = ((h - pp_h)//ps_h) + 1
            w = ((w - pp_w)//ps_w) + 1

        self.dense_layer = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=num_filters[-1]*h*w, out_features=dense_layer),
            identify_aactivation(activation_fn),
            torch.nn.Linear(in_features=dense_layer, out_features=output_size),
        )
        self.use_softmax = use_softmax
    
    def forward(self, x):
        for block in self.conv_blocks:
            x = block(x)
        
        x = self.dense_layer(x)

        if self.use_softmax:
            x = torch.nn.functional.softmax(x, dim=1)
        
        return x


    