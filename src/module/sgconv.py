# Modified from S4: https://github.com/HazyResearch/state-spaces/blob/main/src/models/sequence/ss/s4.py
# We will release the whole codebase upon acceptance.
import math
from functools import partial

import opt_einsum as oe
from src.module.diffusion_attn_unet_1d import Downsample1d, ResConvBlock, Upsample1d
from src.module.integer_encoding import IntegerFourierEmbedding
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

optimized = True

if optimized:
    contract = oe.contract
else:
    contract = torch.einsum


def get_initializer(name, activation=None):
    if activation in [None, 'id', 'identity', 'linear', 'modrelu']:
        nonlinearity = 'linear'
    elif activation in ['relu', 'tanh', 'sigmoid']:
        nonlinearity = activation
    elif activation in ['gelu', 'swish', 'silu']:
        nonlinearity = 'relu'  # Close to ReLU so approximate with ReLU's gain
    else:
        raise NotImplementedError(
            f"get_initializer: activation {activation} not supported")

    if name == 'uniform':
        initializer = partial(
            torch.nn.init.kaiming_uniform_, nonlinearity=nonlinearity)
    elif name == 'normal':
        initializer = partial(torch.nn.init.kaiming_normal_,
                              nonlinearity=nonlinearity)
    elif name == 'xavier':
        initializer = torch.nn.init.xavier_normal_
    elif name == 'zero':
        initializer = partial(torch.nn.init.constant_, val=0)
    elif name == 'one':
        initializer = partial(torch.nn.init.constant_, val=1)
    else:
        raise NotImplementedError(
            f"get_initializer: initializer type {name} not supported")

    return initializer


class modrelu(nn.Module):
    def __init__(self, features):
        # For now we just support square layers
        super(modrelu, self).__init__()
        self.features = features
        self.b = nn.Parameter(torch.Tensor(self.features))
        self.reset_parameters()

    def reset_parameters(self):
        self.b.data.uniform_(-0.01, 0.01)

    def forward(self, inputs):
        norm = torch.abs(inputs)
        biased_norm = norm + self.b
        magnitude = nn.functional.relu(biased_norm)
        phase = torch.sign(inputs)

        return phase * magnitude


class Modrelu(modrelu):
    def reset_parameters(self):
        self.b.data.uniform_(-0.01, 0.01)


class TransposedLinear(nn.Module):
    """ Linear module on the second-to-last dimension
    Assumes shape (B, D, L), where L can be 1 or more axis
    """

    def __init__(self, d_input, d_output, bias=True):
        super().__init__()

        self.weight = nn.Parameter(torch.empty(d_output, d_input))
        # nn.Linear default init
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # nn.init.kaiming_uniform_(self.weight, nonlinearity='linear') # should be equivalent

        if bias:
            self.bias = nn.Parameter(torch.empty(d_output))
            bound = 1 / math.sqrt(d_input)
            nn.init.uniform_(self.bias, -bound, bound)
            setattr(self.bias, "_optim", {"weight_decay": 0.0})
        else:
            self.bias = 0.0

    def forward(self, x):
        num_axis = len(x.shape[2:])  # num_axis in L, for broadcasting bias
        y = contract('b u ..., v u -> b v ...', x, self.weight) + \
            self.bias.view(-1, *[1] * num_axis)
        return y


class TransposedLN(nn.Module):
    """ LayerNorm module over second dimension
    Assumes shape (B, D, L), where L can be 1 or more axis

    This is slow and a dedicated CUDA/Triton implementation shuld provide substantial end-to-end speedup
    """

    def __init__(self, d, scalar=True):
        super().__init__()
        self.scalar = scalar
        if self.scalar:
            self.m = nn.Parameter(torch.zeros(1))
            self.s = nn.Parameter(torch.ones(1))
            setattr(self.m, "_optim", {"weight_decay": 0.0})
            setattr(self.s, "_optim", {"weight_decay": 0.0})
        else:
            self.ln = nn.LayerNorm(d)

    def forward(self, x):
        if self.scalar:
            # calc. stats over D dim / channels
            s, m = torch.std_mean(x, dim=1, unbiased=False, keepdim=True)
            y = (self.s / s) * (x - m + self.m)
        else:
            # move channel to last axis, apply layer_norm, then move channel back to second axis
            _x = self.ln(rearrange(x, 'b d ... -> b ... d'))
            y = rearrange(_x, 'b ... d -> b d ...')
        return y


def Activation(activation=None, size=None, dim=-1):
    if activation in [None, 'id', 'identity', 'linear']:
        return nn.Identity()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'relu':
        return nn.ReLU()
    elif activation == 'gelu':
        return nn.GELU()
    elif activation in ['swish', 'silu']:
        return nn.SiLU()
    elif activation == 'glu':
        return nn.GLU(dim=dim)
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'modrelu':
        return Modrelu(size)
    elif activation == 'sqrelu':
        return nn.SquareReLU()
    elif activation == 'ln':
        return TransposedLN(dim)
    else:
        raise NotImplementedError(
            "hidden activation '{}' is not implemented".format(activation))


def LinearActivation(
        d_input, d_output, bias=True,
        zero_bias_init=False,
        transposed=False,
        initializer=None,
        activation=None,
        activate=False,  # Apply activation as part of this module
        weight_norm=False,
        **kwargs,
):
    """ Returns a linear nn.Module with control over axes order, initialization, and activation """

    # Construct core module
    # linear_cls = partial(nn.Conv1d, kernel_size=1) if transposed else nn.Linear
    linear_cls = TransposedLinear if transposed else nn.Linear
    if activation == 'glu':
        d_output *= 2
    linear = linear_cls(d_input, d_output, bias=bias, **kwargs)

    # Initialize weight
    if initializer is not None:
        get_initializer(initializer, activation)(linear.weight)

    # Initialize bias
    if bias and zero_bias_init:
        nn.init.zeros_(linear.bias)

    # Weight norm
    if weight_norm:
        linear = nn.utils.weight_norm(linear)

    if activate and activation is not None:
        activation = Activation(activation, d_output,
                                dim=1 if transposed else -1)
        linear = nn.Sequential(linear, activation)
    return linear


class Normalization(nn.Module):
    def __init__(
            self,
            d,
            transposed=False,  # Length dimension is -1 or -2
            _name_='layer',
            **kwargs
    ):
        super().__init__()
        self.transposed = transposed
        self._name_ = _name_

        if _name_ == 'layer':
            self.channel = True  # Normalize over channel dimension
            if self.transposed:
                self.norm = TransposedLN(d, **kwargs)
            else:
                self.norm = nn.LayerNorm(d, **kwargs)
        elif _name_ == 'instance':
            self.channel = False
            norm_args = {'affine': False, 'track_running_stats': False}
            norm_args.update(kwargs)
            # (True, True) performs very poorly
            self.norm = nn.InstanceNorm1d(d, **norm_args)
        elif _name_ == 'batch':
            self.channel = False
            norm_args = {'affine': True, 'track_running_stats': True}
            norm_args.update(kwargs)
            self.norm = nn.BatchNorm1d(d, **norm_args)
        elif _name_ == 'group':
            self.channel = False
            self.norm = nn.GroupNorm(1, d, *kwargs)
        elif _name_ == 'none':
            self.channel = True
            self.norm = nn.Identity()
        else:
            raise NotImplementedError

    def forward(self, x):
        # Handle higher dimension logic
        shape = x.shape
        if self.transposed:
            x = rearrange(x, 'b d ... -> b d (...)')
        else:
            x = rearrange(x, 'b ... d -> b (...)d ')

        # The cases of LayerNorm / no normalization are automatically handled in all cases
        # Instance/Batch Norm work automatically with transposed axes
        if self.channel or self.transposed:
            x = self.norm(x)
        else:
            x = x.transpose(-1, -2)
            x = self.norm(x)
            x = x.transpose(-1, -2)

        x = x.view(shape)
        return x

    def step(self, x, **kwargs):
        assert self._name_ in ["layer", "none"]
        if self.transposed:
            x = x.unsqueeze(-1)
        x = self.forward(x)
        if self.transposed:
            x = x.squeeze(-1)
        return x


class GConv(nn.Module):
    requires_length = True

    def __init__(
            self,
            d_model,
            d_state=64,
            l_max=1,
            # Maximum length of sequence. Fine if not provided: the kernel will keep doubling in length until longer than sequence. However, this can be marginally slower if the true length is not a power of 2
            channels=1,  # maps 1-dim to C-dim
            bidirectional=False,
            # Arguments for FF
            activation='gelu',  # activation in between SS and FF
            ln=False,  # Extra normalization
            postact=None,  # activation after FF
            initializer=None,  # initializer on FF
            weight_norm=False,  # weight normalization on FF
            hyper_act=None,  # Use a "hypernetwork" multiplication
            dropout=0.0,
            transposed=True,  # axis ordering (B, L, D) or (B, D, L)
            verbose=False,
            shift=False,
            linear=False,
            mode="cat_randn",
            # SSM Kernel arguments
            **kernel_args,
    ):
        """
        d_state: the dimension of the state, also denoted by N
        l_max: the maximum sequence length, also denoted by L
          if this is not known at model creation, set l_max=1
        channels: can be interpreted as a number of "heads"
        bidirectional: bidirectional
        dropout: standard dropout argument
        transposed: choose backbone axis ordering of (B, L, H) or (B, H, L) [B=batch size, L=sequence length, H=hidden dimension]
        Other options are all experimental and should not need to be configured
        """

        super().__init__()
        self.h = d_model
        self.n = d_state
        self.bidirectional = bidirectional
        self.ln = ln
        self.channels = channels
        self.transposed = transposed
        self.shift = shift
        self.linear = linear
        self.mode = mode
        self.l_max = l_max

        # optional multiplicative modulation GLU-style
        # https://arxiv.org/abs/2002.05202
        self.hyper = hyper_act is not None
        if self.hyper:
            channels *= 2
            self.hyper_activation = Activation(hyper_act)

        self.D = nn.Parameter(torch.randn(channels, self.h))

        if self.bidirectional:
            channels *= 2

        # Pointwise
        if not self.linear:
            self.activation = Activation(activation)
            dropout_fn = nn.Dropout2d if self.transposed else nn.Dropout
            self.dropout = dropout_fn(
                dropout) if dropout > 0.0 else nn.Identity()
            if self.ln:
                self.norm = Normalization(
                    self.h * self.channels, transposed=transposed)
            else:
                self.norm = nn.Identity()

        # position-wise output transform to mix features
        if not self.linear:
            self.output_linear = LinearActivation(
                self.h * self.channels,
                self.h,
                transposed=self.transposed,
                initializer=initializer,
                activation=postact,
                activate=True,
                weight_norm=weight_norm,
            )

        self.init_scale = kernel_args.get('init_scale', 0)
        self.kernel_dim = kernel_args.get('kernel_dim', 64)
        self.num_scales = kernel_args.get(
            'n_scales', 1 + math.ceil(math.log2(l_max / self.kernel_dim)) - self.init_scale)
        if self.num_scales is None:
            self.num_scales = 1 + \
                math.ceil(math.log2(l_max / self.kernel_dim)) - self.init_scale
        self.kernel_list = nn.ParameterList()

        decay_min = kernel_args.get('decay_min', 2)
        decay_max = kernel_args.get('decay_max', 2)

        for _ in range(self.num_scales):
            if 'randn' in mode:
                kernel = nn.Parameter(torch.randn(
                    channels, self.h, self.kernel_dim))
            elif 'cos' in mode:
                kernel = nn.Parameter(torch.cat([torch.cos(torch.linspace(0, 2 * i * math.pi, self.kernel_dim)).expand(
                    channels, 1, self.kernel_dim) for i in range(self.h)], dim=1)[:, torch.randperm(self.h), :])
            else:
                raise ValueError(f"Unknown mode {mode}")
            kernel._optim = {
                'lr': kernel_args.get('lr', 0.001),
            }
            self.kernel_list.append(kernel)

        if 'learnable' in mode:
            self.decay = nn.Parameter(torch.rand(
                self.h) * (decay_max - decay_min) + decay_min)
            if 'fixed' in mode:
                self.decay.requires_grad = False
            else:
                self.decay._optim = {
                    'lr': kernel_args.get('lr', 0.001),
                }
            self.register_buffer('multiplier', torch.tensor(1.0))
        else:
            self.register_buffer('multiplier', torch.linspace(
                decay_min, decay_max, self.h).view(1, -1, 1))

        self.register_buffer('kernel_norm', torch.ones(channels, self.h, 1))
        self.register_buffer('kernel_norm_initialized',
                             torch.tensor(0, dtype=torch.bool))

    # absorbs return_output and transformer src mask
    def forward(self, u, return_kernel=False):
        """
        u: (B H L) if self.transposed else (B L H)
        state: (H N) never needed unless you know what you're doing
        Returns: same shape as u
        """
        if not self.transposed:
            u = u.transpose(-1, -2)
        L = u.size(-1)

        kernel_list = []
        interpolate_mode = 'nearest' if 'nearest' in self.mode else 'linear'
        multiplier = self.multiplier
        if 'sum' in self.mode:
            for i in range(self.num_scales):
                kernel = F.pad(
                    F.interpolate(
                        self.kernel_list[i],
                        scale_factor=2 ** (i + self.init_scale),
                        mode=interpolate_mode,
                    ),
                    (0, self.kernel_dim * 2 ** (self.num_scales - 1 + self.init_scale) -
                     self.kernel_dim * 2 ** (i + self.init_scale)),
                ) * multiplier ** (self.num_scales - i - 1)
                kernel_list.append(kernel)
            k = sum(kernel_list)
        elif 'cat' in self.mode:
            for i in range(self.num_scales):
                kernel = F.interpolate(
                    self.kernel_list[i],
                    scale_factor=2 ** (max(0, i - 1) + self.init_scale),
                    mode=interpolate_mode,
                ) * multiplier ** (self.num_scales - i - 1)
                kernel_list.append(kernel)
            k = torch.cat(kernel_list, dim=-1)
        else:
            raise ValueError(f"Unknown mode {self.mode}")

        if 'learnable' in self.mode:
            k = k * torch.exp(-self.decay.view(1, -1, 1) * torch.log(
                torch.arange(k.size(-1), device=k.device) + 1).view(1, 1, -1))

        if not self.kernel_norm_initialized:
            self.kernel_norm = k.norm(dim=-1, keepdim=True).detach()
            self.kernel_norm_initialized = torch.tensor(
                1, dtype=torch.bool, device=k.device)
            print(f"Kernel norm: {self.kernel_norm.mean()}")
            print(f"Kernel size: {k.size()}")

        if k.size(-1) > L:
            k = k[..., :L]
        elif k.size(-1) < L:
            k = F.pad(k, (0, L - k.size(-1)))

        k = k / self.kernel_norm  # * (L / self.l_max) ** 0.5

        # Convolution
        if self.bidirectional:
            k0, k1 = rearrange(k, '(s c) h l -> s c h l', s=2)
            k = F.pad(k0, (0, L)) + F.pad(k1.flip(-1), (L, 0))
            k_f = torch.fft.rfft(k, n=2 * L)  # (C H L)
        u_f = torch.fft.rfft(u, n=2 * L)  # (B H L)
        # k_f.unsqueeze(-4) * u_f.unsqueeze(-3) # (B C H L)
        y_f = contract('bhl,chl->bchl', u_f, k_f)
        y = torch.fft.irfft(y_f, n=2 * L)[..., :L]  # (B C H L)

        # Compute D term in state space equation - essentially a skip connection
        y = y + contract('bhl,ch->bchl', u, self.D)

        # Reshape to flatten channels
        y = rearrange(y, '... c h l -> ... (c h) l')

        if not self.linear:
            y = self.dropout(self.activation(y))

        if not self.transposed:
            y = y.transpose(-1, -2)

        if not self.linear:
            y = self.norm(y)
            y = self.output_linear(y)

        if return_kernel:
            return y, k
        return y, None

    @property
    def d_state(self):
        return self.h * self.n

    @property
    def d_output(self):
        return self.h

    @property
    def state_to_tensor(self):
        return lambda state: rearrange('... h n -> ... (h n)', state)


class SGConvBlock(nn.Module):
    def __init__(self, input_dim, channels, bidirectional, dropout, l_max=2**13):
        super().__init__()
        self.act1 = nn.GLU()
        self.act2 = nn.GELU()
        self.ff1 = nn.Linear(input_dim, input_dim * 2)
        self.gconv = GConv(
            input_dim, channels=channels, bidirectional=bidirectional, dropout=dropout, transposed=False,
            l_max=l_max, kernel_dim=64
        )
        self.ff2 = nn.Linear(input_dim, input_dim)
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x_in):
        x = self.act1(self.ff1(x_in))
        x = self.gconv(x, return_kernel=False)[0]
        x = self.act2(self.ff2(x))
        x = x + x_in
        x = self.norm(x)
        return x


class GConvStacked(nn.Module):
    """
    d_state: the dimension of the state, also denoted by N
        l_max: the maximum sequence length, also denoted by L
          if this is not known at model creation, set l_max=1
        channels: can be interpreted as a number of "heads"
        bidirectional: bidirectional
        dropout: standard dropout argument
        transposed
    """

    def __init__(self, d_model, channels, bidirectional=True, dropout=0.0, n_layers=10):
        super().__init__()
        self.model = nn.ModuleList(
            [SGConvBlock(d_model, channels, bidirectional, dropout) for _ in range(n_layers)])

    def forward(self, x):
        # x = [batch size, seq len, d_model]
        for layer in self.model:
            x = layer(x)
        return x


class HybridConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim, res_conv_args, gconv_args):
        super().__init__()
        self.res_conv = ResConvBlock(
            c_in=input_dim, c_mid=input_dim*2, c_out=output_dim, transpose=True)
        self.gconv = SGConvBlock(output_dim, **gconv_args)

    def forward(self, x):
        x = self.res_conv(x)
        x = self.gconv(x)
        return x


class HybridConvUnet(nn.Module):
    def __init__(self, input_dim: int, depth: int, conv_args=None, gconv_args=None) -> None:
        super().__init__()
        self.depth = depth

        if conv_args is None:
            conv_args = {}
        if gconv_args is None:
            gconv_args = {
                "channels": 1,
                "bidirectional": True,
                "dropout": 0.0,
                "l_max": 2**16, # 65536
            }

        self.downsample = Downsample1d("cubic", transpose=True)
        self.upsample = Upsample1d("cubic", transpose=True)

        self.down_modules = []
        for i in range(depth):
            output_dim = int(input_dim*1.5)
            if output_dim % 2 == 1:
                output_dim += 1
            self.down_modules.append(HybridConvBlock(
                input_dim, output_dim, conv_args, gconv_args))
            input_dim = output_dim
            print(input_dim)
            gconv_args["l_max"] = gconv_args["l_max"] // 2
            if gconv_args["l_max"] < 2:
                raise ValueError("Initial l_max is too small for the given depth.")

        self.down_modules = nn.ModuleList(self.down_modules)

        self.mid_module = nn.Sequential(
            HybridConvBlock(input_dim, input_dim, conv_args, gconv_args),
        )

        self.up_modules = []
        for i in reversed(range(depth)):
            self.up_modules.append(HybridConvBlock(
                input_dim*2, int(input_dim/1.5), conv_args, gconv_args))
            input_dim = int(input_dim/1.5)
            print(input_dim)
            gconv_args["l_max"] = gconv_args["l_max"] * 2
        self.up_modules = nn.ModuleList(self.up_modules)

    def forward(self, x):
        # x = [batch size, seq len, d_model]
        outputs = []
        for i in range(self.depth):
            x = self.down_modules[i](x)
            outputs.append(x)
            x = self.downsample(x)

        x = self.mid_module(x)

        for i in range(self.depth):
            x = self.upsample(x)
            x = torch.cat([x, outputs[-1-i]], dim=-1)
            x = self.up_modules[i](x)
        return x


class GConvHybrid(nn.Module):
    def __init__(self, data_dim: int, channels: int = 32, dropout: float = 0.0, depth: int = 4, l_max=2**14) -> None:
        super().__init__()
        input_dim = data_dim
        down_modules = []
        up_modules = []
        max_l_max = l_max
        min_l_max = 128

        dim = input_dim

        down_modules = []
        for i in range(depth):
            down_modules.append(ResConvBlock(
                c_in=dim, c_mid=dim*2, c_out=dim, transpose=True))
            down_modules.append(SGConvBlock(
                dim, channels, bidirectional=True, dropout=dropout, l_max=l_max))
            down_modules.append(ResConvBlock(
                c_in=dim, c_mid=dim*2, c_out=dim, transpose=True))
            down_modules.append(SGConvBlock(
                dim, channels, bidirectional=True, dropout=dropout, l_max=l_max))
            if i % 2 == 0:
                down_modules.append(nn.Linear(dim, dim*2)),
                dim *= 2
            down_modules.append(Downsample1d("cubic", transpose=True))
            l_max = max(l_max // 2, min_l_max)
        self.down = nn.ModuleList(down_modules)

        self.mid = nn.Sequential(
            SGConvBlock(dim, channels, bidirectional=True,
                        dropout=dropout, l_max=l_max),
            ResConvBlock(c_in=dim, c_mid=dim*2, c_out=dim, transpose=True),
            SGConvBlock(dim, channels, bidirectional=True,
                        dropout=dropout, l_max=l_max),
            ResConvBlock(c_in=dim, c_mid=dim*2, c_out=dim, transpose=True),
        )

        up_modules = []
        for i in range(depth):
            up_modules.append(Upsample1d("cubic", transpose=True))
            l_max = min(l_max * 2, max_l_max)
            if i % 2 == 0:
                up_modules.append(nn.Linear(dim, dim//2)),
                dim //= 2
            up_modules.append(SGConvBlock(
                dim, channels, bidirectional=True, dropout=dropout, l_max=l_max))
            up_modules.append(ResConvBlock(
                c_in=dim, c_mid=dim*2, c_out=dim, transpose=True))
            up_modules.append(SGConvBlock(
                dim, channels, bidirectional=True, dropout=dropout, l_max=l_max))
            up_modules.append(ResConvBlock(
                c_in=dim, c_mid=dim*2, c_out=dim, transpose=True))
        self.up = nn.ModuleList(up_modules)

    def forward(self, x):
        # x = [batch size, seq len, d_model]
        for layer in self.down:
            x = layer(x)
        x = self.mid(x)
        for layer in self.up:
            x = layer(x)
        return x

class GConvStackedDiffusion(nn.Module):
    """
        d_state: the dimension of the state, also denoted by N
        l_max: the maximum sequence length, also denoted by L
          if this is not known at model creation, set l_max=1
        channels: can be interpreted as a number of "heads"
        bidirectional: bidirectional
        dropout: standard dropout argument
        transposed
    """

    def __init__(self, input_dim, time_emb_dim, model_dim, channels, depth=4, timestep_max_index=1000, cond_dim=0, l_max=2**14):
        super().__init__()
        self.cond_dim = cond_dim

        self.timestep_embed = IntegerFourierEmbedding(
            time_emb_dim, min_index=0, max_index=timestep_max_index)
        self.initial_linear = ResConvBlock(c_in=input_dim+time_emb_dim+cond_dim, c_mid=model_dim, c_out=model_dim, transpose=True)
        self.model = HybridConvUnet(
            input_dim=model_dim, depth=depth, conv_args={}, gconv_args={"channels": channels, "l_max": l_max, "bidirectional": True, "dropout": 0.0}
        )
        self.final_linear = ResConvBlock(c_in=model_dim, c_mid=model_dim, c_out=input_dim, transpose=True)

        self.output_dim = input_dim

    def forward(self, x_in, t, cond=None):
        """
        :param x_in: (batch, seq_len, channels)
        :param t: (), (batch) or (batch, seq_len)
        :param cond: Currently not supported
        :return: (batch, seq_len, channels)
        """

        batch, seq_len, channels = x_in.shape

        # expand t to (batch)
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(batch)
        if t.dim() == 1:
            t = repeat(t, "b -> b t", t=seq_len)
        if t.dim() == 2:
            assert t.shape[0] == batch, "Batch size mismatch between t and x_in"
            assert t.shape[1] == seq_len, "Sequence length mismatch between t and x_in"

        # move t to device
        t = t.to(x_in.device).long()
        t_emb = self.timestep_embed(t)  # (batch, seq_len, 16)

        inputs = [x_in, t_emb]

        if (cond is not None) ^ (self.cond_dim > 0):
            raise ValueError(
                "cond_dim is {}, but cond is {}".format(self.cond_dim, cond))

        if cond is not None:
            if cond.dim() == 2:
                cond = cond.unsqueeze(1)

            cond = rearrange(
                F.interpolate(
                    rearrange(cond, "b s d -> b d s"),
                    x_in.shape[1],
                    mode='linear',
                    align_corners=False), "b d s -> b s d"
            )
            inputs.append(cond)

        x_net = torch.cat(inputs, dim=-1)
        x_net = self.initial_linear(x_net)
        out = self.model(x_net)
        out = self.final_linear(out)
        return out

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def dtype(self):
        return next(self.model.parameters()).dtype


def test_hybridconvunet():
    from torchinfo import summary
    gconv_args = {
        "channels": 1,
        "bidirectional": True,
        "dropout": 0.0,
        "l_max": 2**14,
    }
    model = HybridConvUnet(
        input_dim=64, depth=7, conv_args=None, gconv_args=gconv_args
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    x = torch.randn(1, 2**14, 64)
    # Move the input tensor to the same device as the model
    x = x.to(device)
    y = model(x)
    assert y.shape == x.shape

    summary(model, input_data=(x,), verbose=1)


def test_gconvhybrid():
    from torchinfo import summary
    
    model = GConvHybrid(96, 32,)

    test_x = torch.randn(1, 2048, 128)
    y = model(test_x)

    summary(model, input_data=(test_x,), verbose=1)


if __name__ == "__main__":
    test_hybridconvunet()
