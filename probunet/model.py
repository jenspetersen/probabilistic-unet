import torch
import torch.nn as nn

from probunet.util import make_onehot as make_onehot_segmentation, make_slices, match_to


def is_conv(op):
    conv_types = (nn.Conv1d,
                  nn.Conv2d,
                  nn.Conv3d,
                  nn.ConvTranspose1d,
                  nn.ConvTranspose2d,
                  nn.ConvTranspose3d)
    if type(op) == type and issubclass(op, conv_types):
        return True
    elif type(op) in conv_types:
        return True
    else:
        return False



class ConvModule(nn.Module):

    def __init__(self, *args, **kwargs):

        super(ConvModule, self).__init__()

    def init_weights(self, init_fn, *args, **kwargs):

        class init_(object):

            def __init__(self):
                self.fn = init_fn
                self.args = args
                self.kwargs = kwargs

            def __call__(self, module):
                if is_conv(type(module)):
                    module.weight = self.fn(module.weight, *self.args, **self.kwargs)

        _init_ = init_()
        self.apply(_init_)

    def init_bias(self, init_fn, *args, **kwargs):

        class init_(object):

            def __init__(self):
                self.fn = init_fn
                self.args = args
                self.kwargs = kwargs

            def __call__(self, module):
                if is_conv(type(module)) and module.bias is not None:
                    module.bias = self.fn(module.bias, *self.args, **self.kwargs)

        _init_ = init_()
        self.apply(_init_)



class ConcatCoords(nn.Module):

    def forward(self, input_):

        dim = input_.dim() - 2
        coord_channels = []
        for i in range(dim):
            view = [1, ] * dim
            view[i] = -1
            repeat = list(input_.shape[2:])
            repeat[i] = 1
            coord_channels.append(
                torch.linspace(-0.5, 0.5, input_.shape[i+2])
                .view(*view)
                .repeat(*repeat)
                .to(device=input_.device, dtype=input_.dtype))
        coord_channels = torch.stack(coord_channels).unsqueeze(0)
        repeat = [1, ] * input_.dim()
        repeat[0] = input_.shape[0]
        coord_channels = coord_channels.repeat(*repeat).contiguous()

        return torch.cat([input_, coord_channels], 1)



class InjectionConvEncoder(ConvModule):

    _default_activation_kwargs = dict(inplace=True)
    _default_norm_kwargs = dict()
    _default_conv_kwargs = dict(kernel_size=3, padding=1)
    _default_pool_kwargs = dict(kernel_size=2)
    _default_dropout_kwargs = dict()
    _default_global_pool_kwargs = dict()

    def __init__(self,
                 in_channels=1,
                 out_channels=6,
                 depth=4,
                 injection_depth="last",
                 injection_channels=0,
                 block_depth=2,
                 num_feature_maps=24,
                 feature_map_multiplier=2,
                 activation_op=nn.LeakyReLU,
                 activation_kwargs=None,
                 norm_op=nn.InstanceNorm2d,
                 norm_kwargs=None,
                 norm_depth=0,
                 conv_op=nn.Conv2d,
                 conv_kwargs=None,
                 pool_op=nn.AvgPool2d,
                 pool_kwargs=None,
                 dropout_op=None,
                 dropout_kwargs=None,
                 global_pool_op=nn.AdaptiveAvgPool2d,
                 global_pool_kwargs=None,
                 **kwargs):

        super(InjectionConvEncoder, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.injection_depth = depth - 1 if injection_depth == "last" else injection_depth
        self.injection_channels = injection_channels
        self.block_depth = block_depth
        self.num_feature_maps = num_feature_maps
        self.feature_map_multiplier = feature_map_multiplier

        self.activation_op = activation_op
        self.activation_kwargs = self._default_activation_kwargs
        if activation_kwargs is not None:
            self.activation_kwargs.update(activation_kwargs)

        self.norm_op = norm_op
        self.norm_kwargs = self._default_norm_kwargs
        if norm_kwargs is not None:
            self.norm_kwargs.update(norm_kwargs)
        self.norm_depth = depth if norm_depth == "full" else norm_depth

        self.conv_op = conv_op
        self.conv_kwargs = self._default_conv_kwargs
        if conv_kwargs is not None:
            self.conv_kwargs.update(conv_kwargs)

        self.pool_op = pool_op
        self.pool_kwargs = self._default_pool_kwargs
        if pool_kwargs is not None:
            self.pool_kwargs.update(pool_kwargs)

        self.dropout_op = dropout_op
        self.dropout_kwargs = self._default_dropout_kwargs
        if dropout_kwargs is not None:
            self.dropout_kwargs.update(dropout_kwargs)

        self.global_pool_op = global_pool_op
        self.global_pool_kwargs = self._default_global_pool_kwargs
        if global_pool_kwargs is not None:
            self.global_pool_kwargs.update(global_pool_kwargs)

        for d in range(self.depth):

            in_ = self.in_channels if d == 0 else self.num_feature_maps * (self.feature_map_multiplier**(d-1))
            out_ = self.num_feature_maps * (self.feature_map_multiplier**d)

            if d == self.injection_depth + 1:
                in_ += self.injection_channels

            layers = []
            if d > 0:
                layers.append(self.pool_op(**self.pool_kwargs))
            for b in range(self.block_depth):
                current_in = in_ if b == 0 else out_
                layers.append(self.conv_op(current_in, out_, **self.conv_kwargs))
                if self.norm_op is not None and d < self.norm_depth:
                    layers.append(self.norm_op(out_, **self.norm_kwargs))
                if self.activation_op is not None:
                    layers.append(self.activation_op(**self.activation_kwargs))
                if self.dropout_op is not None:
                    layers.append(self.dropout_op(**self.dropout_kwargs))
            if d == self.depth - 1:
                current_conv_kwargs = self.conv_kwargs.copy()
                current_conv_kwargs["kernel_size"] = 1
                current_conv_kwargs["padding"] = 0
                current_conv_kwargs["bias"] = False
                layers.append(self.conv_op(out_, out_channels, **current_conv_kwargs))

            self.add_module("encode_{}".format(d), nn.Sequential(*layers))

        if self.global_pool_op is not None:
            self.add_module("global_pool", self.global_pool_op(1, **self.global_pool_kwargs))

    def forward(self, x, injection=None):

        for d in range(self.depth):
            x = self._modules["encode_{}".format(d)](x)
            if d == self.injection_depth and self.injection_channels > 0:
                injection = match_to(injection, x, self.injection_channels)
                x = torch.cat([x, injection], 1)
        if hasattr(self, "global_pool"):
            x = self.global_pool(x)

        return x


class InjectionConvEncoder3D(InjectionConvEncoder):

    def __init__(self, *args, **kwargs):

        update_kwargs = dict(
                norm_op=nn.InstanceNorm3d,
                conv_op=nn.Conv3d,
                pool_op=nn.AvgPool3d,
                global_pool_op=nn.AdaptiveAvgPool3d
            )

        for (arg, val) in update_kwargs.items():
            if arg not in kwargs: kwargs[arg] = val

        super(InjectionConvEncoder3D, self).__init__(*args, **kwargs)



class InjectionUNet(ConvModule):

    def __init__(
        self,
        depth=5,
        in_channels=4,
        out_channels=4,
        kernel_size=3,
        dilation=1,
        num_feature_maps=24,
        block_depth=2,
        num_1x1_at_end=3,
        injection_channels=3,
        injection_at="end",
        activation_op=nn.LeakyReLU,
        activation_kwargs=None,
        pool_op=nn.AvgPool2d,
        pool_kwargs=dict(kernel_size=2),
        dropout_op=None,
        dropout_kwargs=None,
        norm_op=nn.InstanceNorm2d,
        norm_kwargs=None,
        conv_op=nn.Conv2d,
        conv_kwargs=None,
        upconv_op=nn.ConvTranspose2d,
        upconv_kwargs=None,
        output_activation_op=None,
        output_activation_kwargs=None,
        return_bottom=False,
        coords=False,
        coords_dim=2,
        **kwargs
    ):

        super(InjectionUNet, self).__init__(**kwargs)

        self.depth = depth
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = (self.kernel_size + (self.kernel_size-1) * (self.dilation-1)) // 2
        self.num_feature_maps = num_feature_maps
        self.block_depth = block_depth
        self.num_1x1_at_end = num_1x1_at_end
        self.injection_channels = injection_channels
        self.injection_at = injection_at
        self.activation_op = activation_op
        self.activation_kwargs = {} if activation_kwargs is None else activation_kwargs
        self.pool_op = pool_op
        self.pool_kwargs = {} if pool_kwargs is None else pool_kwargs
        self.dropout_op = dropout_op
        self.dropout_kwargs = {} if dropout_kwargs is None else dropout_kwargs
        self.norm_op = norm_op
        self.norm_kwargs = {} if norm_kwargs is None else norm_kwargs
        self.conv_op = conv_op
        self.conv_kwargs = {} if conv_kwargs is None else conv_kwargs
        self.upconv_op = upconv_op
        self.upconv_kwargs = {} if upconv_kwargs is None else upconv_kwargs
        self.output_activation_op = output_activation_op
        self.output_activation_kwargs = {} if output_activation_kwargs is None else output_activation_kwargs
        self.return_bottom = return_bottom
        if not coords:
            self.coords = [[], []]
        elif coords is True:
            self.coords = [list(range(depth)), []]
        else:
            self.coords = coords
        self.coords_dim = coords_dim

        self.last_activations = None

        # BUILD ENCODER
        for d in range(self.depth):

            block = []
            if d > 0:
                block.append(self.pool_op(**self.pool_kwargs))

            for i in range(self.block_depth):

                # bottom block fixed to have depth 1
                if d == self.depth - 1 and i > 0:
                    continue

                out_size = self.num_feature_maps * 2**d
                if d == 0 and i == 0:
                    in_size = self.in_channels
                elif i == 0:
                    in_size = self.num_feature_maps * 2**(d - 1)
                else:
                    in_size = out_size

                # check for coord appending at this depth
                if d in self.coords[0] and i == 0:
                    block.append(ConcatCoords())
                    in_size += self.coords_dim

                block.append(self.conv_op(in_size,
                                          out_size,
                                          self.kernel_size,
                                          padding=self.padding,
                                          dilation=self.dilation,
                                          **self.conv_kwargs))
                if self.dropout_op is not None:
                    block.append(self.dropout_op(**self.dropout_kwargs))
                if self.norm_op is not None:
                    block.append(self.norm_op(out_size, **self.norm_kwargs))
                block.append(self.activation_op(**self.activation_kwargs))

            self.add_module("encode-{}".format(d), nn.Sequential(*block))

        # BUILD DECODER
        for d in reversed(range(self.depth)):

            block = []

            for i in range(self.block_depth):

                # bottom block fixed to have depth 1
                if d == self.depth - 1 and i > 0:
                    continue

                out_size = self.num_feature_maps * 2**(d)
                if i == 0 and d < self.depth - 1:
                    in_size = self.num_feature_maps * 2**(d+1)
                elif i == 0 and self.injection_at == "bottom":
                    in_size = out_size + self.injection_channels
                else:
                    in_size = out_size

                # check for coord appending at this depth
                if d in self.coords[0] and i == 0 and d < self.depth - 1:
                    block.append(ConcatCoords())
                    in_size += self.coords_dim

                block.append(self.conv_op(in_size,
                                          out_size,
                                          self.kernel_size,
                                          padding=self.padding,
                                          dilation=self.dilation,
                                          **self.conv_kwargs))
                if self.dropout_op is not None:
                    block.append(self.dropout_op(**self.dropout_kwargs))
                if self.norm_op is not None:
                    block.append(self.norm_op(out_size, **self.norm_kwargs))
                block.append(self.activation_op(**self.activation_kwargs))

            if d > 0:
                block.append(self.upconv_op(out_size,
                                            out_size // 2,
                                            self.kernel_size,
                                            2,
                                            padding=self.padding,
                                            dilation=self.dilation,
                                            output_padding=1,
                                            **self.upconv_kwargs))

            self.add_module("decode-{}".format(d), nn.Sequential(*block))

        if self.injection_at == "end":
            out_size += self.injection_channels
        in_size = out_size
        for i in range(self.num_1x1_at_end):
            if i == self.num_1x1_at_end - 1:
                out_size = self.out_channels
            current_conv_kwargs = self.conv_kwargs.copy()
            current_conv_kwargs["bias"] = True
            self.add_module("reduce-{}".format(i), self.conv_op(in_size, out_size, 1, **current_conv_kwargs))
            if i != self.num_1x1_at_end - 1:
                self.add_module("reduce-{}-nonlin".format(i), self.activation_op(**self.activation_kwargs))
        if self.output_activation_op is not None:
            self.add_module("output-activation", self.output_activation_op(**self.output_activation_kwargs))

    def reset(self):

        self.last_activations = None

    def forward(self, x, injection=None, reuse_last_activations=False, store_activations=False):

        if self.injection_at == "bottom":  # not worth it for now
            reuse_last_activations = False
            store_activations = False

        if self.last_activations is None or reuse_last_activations is False:

            enc = [x]

            for i in range(self.depth - 1):
                enc.append(self._modules["encode-{}".format(i)](enc[-1]))

            bottom_rep = self._modules["encode-{}".format(self.depth - 1)](enc[-1])

            if self.injection_at == "bottom" and self.injection_channels > 0:
                injection = match_to(injection, bottom_rep, (0, 1))
                bottom_rep = torch.cat((bottom_rep, injection), 1)

            x = self._modules["decode-{}".format(self.depth - 1)](bottom_rep)

            for i in reversed(range(self.depth - 1)):
                x = self._modules["decode-{}".format(i)](torch.cat((enc[-(self.depth - 1 - i)], x), 1))

            if store_activations:
                self.last_activations = x.detach()

        else:

            x = self.last_activations

        if self.injection_at == "end" and self.injection_channels > 0:
            injection = match_to(injection, x, (0, 1))
            x = torch.cat((x, injection), 1)

        for i in range(self.num_1x1_at_end):
            x = self._modules["reduce-{}".format(i)](x)
        if self.output_activation_op is not None:
            x = self._modules["output-activation"](x)

        if self.return_bottom and not reuse_last_activations:
            return x, bottom_rep
        else:
            return x



class InjectionUNet3D(InjectionUNet):

    def __init__(self, *args, **kwargs):

        update_kwargs = dict(
                pool_op=nn.AvgPool3d,
                norm_op=nn.InstanceNorm3d,
                conv_op=nn.Conv3d,
                upconv_op=nn.ConvTranspose3d,
                coords_dim=3
            )

        for (arg, val) in update_kwargs.items():
            if arg not in kwargs: kwargs[arg] = val

        super(InjectionUNet3D, self).__init__(*args, **kwargs)



class ProbabilisticSegmentationNet(ConvModule):

    def __init__(self,
                 in_channels=4,
                 out_channels=4,
                 num_feature_maps=24,
                 latent_size=3,
                 depth=5,
                 latent_distribution=torch.distributions.Normal,
                 task_op=InjectionUNet3D,
                 task_kwargs=None,
                 prior_op=InjectionConvEncoder3D,
                 prior_kwargs=None,
                 posterior_op=InjectionConvEncoder3D,
                 posterior_kwargs=None,
                 **kwargs):

        super(ProbabilisticSegmentationNet, self).__init__(**kwargs)

        self.task_op = task_op
        self.task_kwargs = {} if task_kwargs is None else task_kwargs
        self.prior_op = prior_op
        self.prior_kwargs = {} if prior_kwargs is None else prior_kwargs
        self.posterior_op = posterior_op
        self.posterior_kwargs = {} if posterior_kwargs is None else posterior_kwargs

        default_task_kwargs = dict(
            in_channels=in_channels,
            out_channels=out_channels,
            num_feature_maps=num_feature_maps,
            injection_size=latent_size,
            depth=depth
        )

        default_prior_kwargs = dict(
            in_channels=in_channels,
            num_feature_maps=num_feature_maps,
            z_dim=latent_size,
            depth=depth
        )

        default_posterior_kwargs = dict(
            in_channels=in_channels+out_channels,
            num_feature_maps=num_feature_maps,
            z_dim=latent_size,
            depth=depth
        )

        default_task_kwargs.update(self.task_kwargs)
        self.task_kwargs = default_task_kwargs
        default_prior_kwargs.update(self.prior_kwargs)
        self.prior_kwargs = default_prior_kwargs
        default_posterior_kwargs.update(self.posterior_kwargs)
        self.posterior_kwargs = default_posterior_kwargs

        self.latent_distribution = latent_distribution
        self._prior = None
        self._posterior = None

        self.make_modules()

    def make_modules(self):

        if type(self.task_op) == type:
            self.add_module("task_net", self.task_op(**self.task_kwargs))
        else:
            self.add_module("task_net", self.task_op)
        if type(self.prior_op) == type:
            self.add_module("prior_net", self.prior_op(**self.prior_kwargs))
        else:
            self.add_module("prior_net", self.prior_op)
        if type(self.posterior_op) == type:
            self.add_module("posterior_net", self.posterior_op(**self.posterior_kwargs))
        else:
            self.add_module("posterior_net", self.posterior_op)

    @property
    def prior(self):
        return self._prior

    @property
    def posterior(self):
        return self._posterior

    @property
    def last_activations(self):
        return self.task_net.last_activations

    def train(self, mode=True):

        super(ProbabilisticSegmentationNet, self).train(mode)
        self.reset()

    def reset(self):

        self.task_net.reset()
        self._prior = None
        self._posterior = None

    def forward(self, input_, seg=None, make_onehot=True, make_onehot_classes=None, newaxis=False):
        """Forward pass includes reparametrization sampling during training, otherwise it'll just take the prior mean."""

        self.encode_prior(input_)
        if self.training:
            self.encode_posterior(input_, seg, make_onehot, make_onehot_classes, newaxis)
            sample = self.posterior.rsample()
        else:
            sample = self.prior.loc
        return self.task_net(input_, sample, store_activations=not self.training)

    def encode_prior(self, input_):

        rep = self.prior_net(input_)
        if isinstance(rep, tuple):
            mean, logvar = rep
        elif torch.is_tensor(rep):
            mean, logvar = torch.split(rep, rep.shape[1] // 2, dim=1)
        self._prior = self.latent_distribution(mean, logvar.mul(0.5).exp())
        return self._prior

    def encode_posterior(self, input_, seg, make_onehot=True, make_onehot_classes=None, newaxis=False):

        if make_onehot:
            if make_onehot_classes is None:
                make_onehot_classes = tuple(range(self.posterior_net.in_channels - input_.shape[1]))
            seg = make_onehot_segmentation(seg, make_onehot_classes, newaxis=newaxis)
        rep = self.posterior_net(torch.cat((input_, seg.float()), 1))
        if isinstance(rep, tuple):
            mean, logvar = rep
        elif torch.is_tensor(rep):
            mean, logvar = torch.split(rep, rep.shape[1] // 2, dim=1)
        self._posterior = self.latent_distribution(mean, logvar.mul(0.5).exp())
        return self._posterior

    def sample_prior(self, N=1, out_device=None, input_=None):
        """Draw multiple samples from the current prior.
        
        * input_ is required if no activations are stored in task_net.
        * If input_ is given, prior will automatically be encoded again.
        * Returns either a single sample or a list of samples.

        """

        if out_device is None:
            if self.last_activations is not None:
                out_device = self.last_activations.device
            elif input_ is not None:
                out_device = input_.device
            else:
                out_device = next(self.task_net.parameters()).device
        with torch.no_grad():
            if self.prior is None or input_ is not None:
                self.encode_prior(input_)
            result = []
            if input_ is not None:
                result.append(self.task_net(input_, self.prior.sample(), reuse_last_activations=False, store_activations=True).to(device=out_device))
            while len(result) < N:
                result.append(self.task_net(input_,
                                            self.prior.sample(),
                                            reuse_last_activations=self.last_activations is not None,
                                            store_activations=False).to(device=out_device))
            if N == 1:
                return result[0]
            else:
                return result

    def reconstruct(self, sample=None, use_posterior_mean=True, out_device=None, input_=None):
        """Reconstruct a sample or the current posterior mean. Will not compute gradients!"""

        if self.posterior is None and sample is None:
            raise ValueError("'posterior' is currently None. Please pass an input and a segmentation first.")
        if out_device is None:
            out_device = next(self.task_net.parameters()).device
        if sample is None:
            if use_posterior_mean:
                sample = self.posterior.loc
            else:
                sample = self.posterior.sample()
        else:
            sample = sample.to(next(self.task_net.parameters()).device)
        with torch.no_grad():
            return self.task_net(input_, sample, reuse_last_activations=True).to(device=out_device)

    def kl_divergence(self):
        """Compute current KL, requires existing prior and posterior."""

        if self.posterior is None or self.prior is None:
            raise ValueError("'prior' and 'posterior' must not be None, but prior={} and posterior={}".format(self.prior, self.posterior))
        return torch.distributions.kl_divergence(self.posterior, self.prior).sum()

    def elbo(self, seg, input_=None, nll_reduction="sum", beta=1.0, make_onehot=True, make_onehot_classes=None, newaxis=False):
        """Compute the ELBO with seg as ground truth.

        * Prior is expected and will not be encoded.
        * If input_ is given, posterior will automatically be encoded.
        * Either input_ or stored activations must be available.

        """

        if self.last_activations is None:
            raise ValueError("'last_activations' is currently None. Please pass an input first.")
        if input_ is not None:
            with torch.no_grad():
                self.encode_posterior(input_, seg, make_onehot=make_onehot, make_onehot_classes=make_onehot_classes, newaxis=newaxis)
        if make_onehot and newaxis:
            pass  # seg will already be (B x SPACE)
        elif make_onehot and not newaxis:
            seg = seg[:, 0]  # in this case seg will hopefully be (B x 1 x SPACE)
        else:
            seg = torch.argmax(seg, 1, keepdim=False)  # seg is already onehot
        kl = self.kl_divergence()
        nll = nn.NLLLoss(reduction=nll_reduction)(self.reconstruct(sample=None, use_posterior_mean=True, out_device=None), seg.long())
        return - (beta * nll + kl)
