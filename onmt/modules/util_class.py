""" Misc classes """
import torch
import torch.nn as nn


# At the moment this class is only used by embeddings.Embeddings look-up tables
class Elementwise(nn.ModuleList):
    """
    A simple network container.
    Parameters are a list of modules.
    Inputs are a 3d Tensor whose last dimension is the same length
    as the list.
    Outputs are the result of applying modules to inputs elementwise.
    An optional merge parameter allows the outputs to be reduced to a
    single Tensor.
    """

    def __init__(self, merge=None, *args):
        assert merge in [None, 'first', 'concat', 'sum', 'mlp']
        self.merge = merge
        super(Elementwise, self).__init__(*args)

    def forward(self, inputs):
        inputs_ = [feat.squeeze(2) for feat in inputs.split(1, dim=2)]
        assert len(self) == len(inputs_)
        outputs = [f(x) for f, x in zip(self, inputs_)]
        if self.merge == 'first':
            return outputs[0]
        elif self.merge == 'concat' or self.merge == 'mlp':
            return torch.cat(outputs, 2)
        elif self.merge == 'sum':
            return sum(outputs)
        else:
            return outputs


class Cast(nn.Module):
    """
    Basic layer that casts its input to a specific data type. The same tensor
    is returned if the data type is already correct.
    """

    def __init__(self, dtype):
        super(Cast, self).__init__()
        self._dtype = dtype

    def forward(self, x):
        return x.to(self._dtype)


class PoincareReparametrize(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.phi_dir = nn.Linear(in_dim,out_dim)
        self.phi_norm = nn.Linear(in_dim,1)
    # x: [batch_]
    def forward(self, x):
        """
        Args:
            x (FloatTensor): batch of vectors
                 ``(batch, vec_size)``.

        Returns:
            * outs: output from the transforming
              ``(batch, out_dim)``.
        """
        v_bar  = self.phi_dir(x)
        p_bar = self.phi_norm(x)
        v = v_bar / torch.norm(v_bar, dim = -1).unsqueeze(-1)
        p = nn.functional.sigmoid(p_bar)

        return p*v

    @staticmethod
    def poincare_dist(u, e):
        """
        Args:
            u (FloatTensor): batch of vectors
                 ``(batch, vec_size)``.
            e (FloatTensor): batch of vectors
                 ``(vocab_size, vec_size)``.

        Returns:
            * outs: output from the transforming
              ``(batch, 1)``.
        """
        #euclidean norm
        squnorm = torch.sum(u * u, dim=-1)
        res = []
        for e_slice in e.split(1,0):
            sqvnorm = torch.sum(e_slice * e_slice, dim=-1)
            sqdist = torch.sum(torch.pow(u - e_slice, 2), dim=-1)
            #fraction
            x = sqdist / ((1 - squnorm) * (1 - sqvnorm)) * 2 + 1
            # arcosh
            z = torch.sqrt(torch.pow(x, 2) - 1)
            res.append(torch.log(x + z))

        return torch.stack(res,1)
