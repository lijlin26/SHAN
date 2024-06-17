"""Graph decoders."""
import manifolds
import torch.nn as nn
from layers.layers import GraphConvolution, Linear
from dgl.nn.pytorch import GATConv

class Decoder(nn.Module):
    """
    Decoder abstract class for node classification tasks.
    """

    def __init__(self, c):
        super(Decoder, self).__init__()
        self.c = c

    def decode(self, x, adj):
        if self.decode_adj:
            probs = self.cls(adj, x)
            probs = probs.mean(dim=1)
        else:
            probs = self.cls(x)
        return probs


class GCNDecoder(Decoder):
    """
    Graph Convolution Decoder.
    """
    def __init__(self, c, args):
        super(GCNDecoder, self).__init__(c)
        act = lambda x: x
        self.cls = GraphConvolution(args.dim, args.n_classes, args.dropout, act, args.bias)
        self.decode_adj = True

    def decode(self, x, adj):
        h = self.manifold.proj_tan0(self.manifold.logmap0(x, c=self.c), c=self.c)

        return super(GCNDecoder, self).decode(h, adj)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, c={}'.format(
                self.input_dim, self.output_dim, self.bias, self.c
        )

class GATDecoder(Decoder):
    """
    Graph Attention Decoder.
    """

    def __init__(self, c, args):
        super(GATDecoder, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        self.cls = GATConv(args.dim * args.decoder_heads,  args.n_classes, feat_drop = args.dropout, attn_drop = args.atten_drop, residual=args.decoder_residual, num_heads = args.decoder_heads, model_name='decoder')
        
        self.decode_adj = True

    def decode(self, x, adj):
        h = self.manifold.proj_tan0(self.manifold.logmap0(x, c=self.c), c=self.c)

        return super(GATDecoder, self).decode(h, adj)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, c={}'.format(
                self.input_dim, self.output_dim, self.bias, self.c
        )



class LinearDecoder(Decoder):
    """
    MLP Decoder for Hyperbolic/Euclidean node classification models.
    """

    def __init__(self, c, args):
        super(LinearDecoder, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        self.input_dim = args.dim * args.encoder_heads
        self.output_dim = args.n_classes
        self.bias = args.bias
        self.cls = Linear(self.input_dim , self.output_dim, args.dropout, lambda x: x, self.bias)
        self.decode_adj = False

    def decode(self, x, adj):
        h = self.manifold.proj_tan0(self.manifold.logmap0(x, c=self.c), c=self.c)
        return super(LinearDecoder, self).decode(h, adj)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, c={}'.format(
                self.input_dim, self.output_dim, self.bias, self.c
        )
