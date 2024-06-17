import torch.nn as nn
import torch

import manifolds
import models.encoders as encoders
# from models.decoders import GATDecoder, LinearDecoder
import models.decoders as decoders



class SHAN(nn.Module):
    def __init__(self, args) -> None:
        super(SHAN, self).__init__()
        self.manifold_name = args.manifold
        self.c = nn.Parameter(torch.Tensor([1.]))
        self.manifold = getattr(manifolds, args.manifold)()
        if self.manifold_name == 'Hyperboloid':
            args.feat_dim = args.feat_dim + 1

        self.encoder = getattr(encoders, 'HGCN')(self.c, args)


        if args.decoder == 'linear':
            self.decoder = getattr(decoders, 'LinearDecoder')(self.c, args)
        elif args.decoder == 'gat':
            self.decoder = getattr(decoders, 'GATDecoder')(self.c, args)
        else:
            raise NotImplementedError
        

        self.attn_simplex = nn.Parameter(torch.FloatTensor(size=(args.dim * args.encoder_heads, 1)))
        self.attn_complex = nn.Parameter(torch.FloatTensor(size=(args.dim * args.encoder_heads, 1)))
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.attn_simplex, gain=gain)
        nn.init.xavier_normal_(self.attn_complex, gain=gain)

    def forward(self, input):
        features, graph, graph_homos = input
        if self.manifold_name ==  'Hyperboloid':
            o = torch.zeros_like(features)
            features = torch.cat([o[:, 0:1], features], dim=1)
        h = self.encoder.encode(features, graph)
        h_homos = []

        if len(graph_homos):
            for graph_homo in graph_homos:
                h_homo_temp = self.encoder.encode(features, graph_homo)
                h_homos.append(h_homo_temp.unsqueeze(dim=0))

            h_homos = torch.cat(h_homos, dim=0)

            h_homo = self.simplex_level_atten(h_homos)
            h = self.complex_level_atten(h, h_homo)

        h = self.decoder.decode(h, graph)
        return h


    def simplex_level_atten(self, h_homos):
        h_homos = self.manifold.proj_tan0(self.manifold.logmap0(h_homos, c=self.c), c=self.c)

        h_homos_pre = h_homos.max(dim=1).values
        h_homos_atten = torch.softmax(torch.mm(h_homos_pre, self.attn_simplex), dim=0).unsqueeze(dim=-1)
        h_homos = torch.mul(h_homos, h_homos_atten)
        h_homo = h_homos.sum(dim=0)

        
        h_homo = self.manifold.proj(self.manifold.expmap0(h_homo, c=self.c), c=self.c)
        return h_homo


    def complex_level_atten(self, h, h_homo):
        h = self.manifold.proj_tan0(self.manifold.logmap0(h, c=self.c), c=self.c)
        h_homo = self.manifold.proj_tan0(self.manifold.logmap0(h_homo, c=self.c), c=self.c)

        h_pre = h.max(dim=0).values.unsqueeze(dim=0)
        h_homo_pre = h_homo.max(dim=0).values.unsqueeze(dim=0)
        h_pre = torch.cat((h_pre, h_homo_pre), dim=0)
        h_atten = torch.softmax(torch.mm(h_pre, self.attn_complex), dim=0)
        h = h * h_atten[0] + h_homo*h_atten[1]

        h = self.manifold.proj(self.manifold.expmap0(h, c=self.c), c=self.c)

        return h



