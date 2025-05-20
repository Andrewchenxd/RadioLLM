import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        
    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1./sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        
        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top): # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k)) # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     M_top, :] # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1)) # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else: # use mask
            assert(L_Q == L_V) # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1) # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V])/L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2,1)
        keys = keys.transpose(2,1)
        values = values.transpose(2,1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item() # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item() # c*ln(L_q) 

        U_part = U_part if U_part<L_K else L_K
        u = u if u<L_Q else L_Q
        
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u) 

        # add scale factor
        scale = self.scale or 1./sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)
        
        return context.transpose(2,1).contiguous(), attn


class FreqAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FreqAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        queries_freq =torch.abs(torch.fft.fft(queries))
        keys_freq =torch.abs(torch.fft.fft(keys))
        values_freq = torch.abs(torch.fft.fft(values))

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)
        scores_top_freq, index_freq = self._prob_QK(queries_freq, keys_freq, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        context_freq = self._get_initial_context(values_freq, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)
        context_freq, _ = self._update_context(context_freq, values_freq, scores_top_freq, index_freq, L_Q, attn_mask)
        context_freq=torch.abs(torch.fft.ifft(context_freq))
        return context.transpose(2, 1).contiguous(),context_freq.transpose(2, 1).contiguous(), attn

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, 
                 d_keys=None, d_values=None, mix=False,attn='prob'):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix
        self.attn=attn
        if self.attn=='freq':
            self.pro_linear=nn.Linear(d_values * n_heads,d_values * n_heads)
            self.pro_linear_freq = nn.Linear(d_values * n_heads, d_values * n_heads)

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        if self.attn!='freq':
            out, attn = self.inner_attention(
                queries,
                keys,
                values,
                attn_mask
            )
            if self.mix:
                out = out.transpose(2, 1).contiguous()
            out = out.view(B, L, -1)
        else:
            out,out_freq, attn = self.inner_attention(
                queries,
                keys,
                values,
                attn_mask
            )
            if self.mix:
                out = out.transpose(2, 1).contiguous()
                out_freq = out_freq.transpose(2, 1).contiguous()
            out = out.view(B, L, -1)
            out_freq=out_freq.view(B, L, -1)
            out=self.pro_linear(out)
            out_freq = self.pro_linear_freq(out_freq)
            out=out+out_freq+values.view(B, L, -1)


        return self.out_projection(out), attn

class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))

        return self.norm3(x+y)



class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)

        return x
    
    

    
class Bottleneck(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet1d(nn.Module):
    def __init__(self, block=Bottleneck, layers=[2,2,2,2],numclass=11):
        super(ResNet1d, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv1d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.mlp_head=nn.Sequential(nn.Linear(512,256),
                                    nn.GELU(),
                                    nn.Dropout(0.1),
                                    nn.Linear(256,numclass))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool1d(out, out.size(2))
        out = out.view(out.size(0), -1)
        out=self.mlp_head(out)
        return out



class ResNet1d_fuse(nn.Module):
    def __init__(self, block=Bottleneck, layers=[2,2,2,2],numclass=11):
        super(ResNet1d_fuse, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv1d(2, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(32)

        self.conv_n1 = nn.Conv1d(2, 32, kernel_size=1, stride=1)
        self.conv_n2 = nn.Conv1d(32, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn_n1 = nn.BatchNorm1d(32)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.mlp_head=nn.Sequential(nn.Linear(512,256),
                                    nn.GELU(),
                                    nn.Dropout(0.1),
                                    nn.Linear(256,numclass))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x_pred,x_org):
        out = F.relu(self.bn1(self.conv1(x_pred)))
        out_n=F.relu(self.bn_n1(self.conv_n2(self.conv_n1(x_org))))
        out=torch.concatenate([out,out_n],1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool1d(out, out.size(2))
        out = out.view(out.size(0), -1)
        out=self.mlp_head(out)
        return out
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model,dmodel2):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        # self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
        #                             kernel_size=3, padding=padding, padding_mode='circular')
        self.conv1 = nn.Conv1d(c_in, dmodel2, 3,
                               padding=padding , padding_mode='circular')
        self.bn1 = nn.BatchNorm1d(dmodel2)
        self.gelu = nn.GELU()

        self.conv2 = nn.Conv1d(dmodel2, d_model, 3,
                               padding=padding , padding_mode='circular')
        self.bn2 = nn.BatchNorm1d(d_model)

        self.conv3 = nn.Conv1d(c_in, d_model, 7,
                               padding=3 , padding_mode='replicate')
        self.bn3 = nn.BatchNorm1d(d_model)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')

    def forward(self, x):
        x=x.permute(0, 2, 1)
        out = self.gelu(self.bn1(self.conv1(x)))
        out = self.gelu(self.bn2(self.conv2(out)))
        out = out + self.gelu(self.bn3(self.conv3(x)))
        out=out.transpose(1,2)
        # out = self.tokenConv(x.permute(0, 2, 1)).transpose(1,2)

        return out

class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()

class Diff_FixedEmbedding(nn.Module):
    def __init__(self, out_len, d_model,loop=0,max_diffusion_steps = 1000.0):
        super(Diff_FixedEmbedding, self).__init__()

        w = torch.zeros(1,out_len, d_model).float()
        w.require_grad = False

        position = torch.arange(0, out_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(max_diffusion_steps) / d_model)).exp()

        w[0,:, 0::2] = torch.sin(loop*position * div_term)
        w[0,:, 1::2] = torch.cos(loop*position * div_term)

        self.emb = nn.Parameter(w)

    def forward(self):
        return self.emb

class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(LearnablePositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        self.pe = nn.Parameter(torch.zeros(
            1, max_len, d_model), requires_grad=True)

        pe = torch.zeros(max_len, d_model).float()
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.pe.data.copy_(pe.float())
        del pe

    def forward(self, x, offset=0):
        return self.pe[:, offset:offset+x.size(1)]

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding, self).__init__()
        if d_model<=128:
            d_model2=256
        else:
            d_model2=d_model*2
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model,dmodel2=d_model2)
        self.position_embedding =LearnablePositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark)
        x = self.value_embedding(x) + self.position_embedding(x)

        return self.dropout(x)

class DataEmbedding_diff(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1,out_len=128,loop=0,max_diffusion_steps = 1000 ):
        super(DataEmbedding_diff, self).__init__()
        if d_model<=128:
            d_model2=256
        else:
            d_model2=d_model*2
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model,dmodel2=d_model2)
        # self.position_embedding = PositionalEmbedding(d_model=d_model)
        # self.position_embedding =nn.Parameter(torch.randn(1, out_len, d_model))
        self.dropout = nn.Dropout(p=dropout)
        # 假设扩散过程总共有max_diffusion_steps步
        self.diff_embed = Diff_FixedEmbedding(out_len, d_model,loop=loop,max_diffusion_steps = max_diffusion_steps)
    def forward(self, x):
        # x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark)
        x = self.value_embedding(x) + self.diff_embed()

        return self.dropout(x)
    
class ResNet1d_enc(nn.Module):
    def __init__(self, block=Bottleneck, layers=[2,2,2,2],numclass=11):
        super(ResNet1d_enc, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv1d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.mlp_head=nn.Sequential(nn.Linear(512 , 256),
                                    nn.GELU(),
                                    nn.Dropout(0.1),
                                    nn.Linear(256 , numclass))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # x = x.unsqueeze(1)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool1d(out, out.size(2))
        out = out.view(out.size(0), -1)
        out=self.mlp_head(out)
        return out

class High_freq_conv(nn.Module):
    def __init__(self, d_model,out_channel, patch_len, stride, dropout):
        super(High_freq_conv, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = ReplicationPad1d((0, stride))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding =nn.Conv2d(1, d_model, kernel_size=(1, patch_len), stride=(1,stride), padding=(0, stride), bias=False)

        self.high_freq_exter=high_freq_extract(in_channels=d_model,out_channels=d_model*2)
        self.high_freq_exter2 = high_freq_extract(in_channels=d_model, out_channels=out_channel * 2)
        self.pool=nn.MaxPool1d(kernel_size=2)
        self.drop=nn.Dropout(dropout)

    def forward(self, x):
        x = self.padding_patch_layer(x)
        x=torch.unsqueeze(x,1)
        x = self.value_embedding(x)
        x = self.drop(x)
        b,n,d,l=x.shape
        x = einops.rearrange(x, 'b d n l -> b (n d) l')
        x=self.high_freq_exter(x)
        x=self.pool(x)
        x = self.high_freq_exter2(x)
        x = self.pool(x)
        return x


class High_freq_conv_3layer(nn.Module):
    def __init__(self, d_model,out_channel, patch_len, stride, dropout):
        super(High_freq_conv_3layer, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = ReplicationPad1d((0, stride))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding =nn.Conv2d(1, d_model, kernel_size=(1, patch_len), stride=(1,stride), padding=(0, stride), bias=False)

        self.high_freq_exter=high_freq_extract(in_channels=d_model,out_channels=d_model*2)
        self.high_freq_exter2 = high_freq_extract(in_channels=d_model, out_channels=out_channel * 2)
        self.high_freq_exter3 = high_freq_extract(in_channels=out_channel, out_channels=out_channel * 2)
        self.pool=nn.MaxPool1d(kernel_size=2)
        self.drop=nn.Dropout(dropout)

    def forward(self, x):
        x = self.padding_patch_layer(x)
        x=torch.unsqueeze(x,1)
        x = self.value_embedding(x)
        x = self.drop(x)
        b,n,d,l=x.shape
        x = einops.rearrange(x, 'b d n l -> b (n d) l')
        
        x=self.high_freq_exter(x)
        x=self.pool(x)
        x = self.high_freq_exter2(x)
        x = self.pool(x)
        x = self.high_freq_exter3(x)
        x = self.pool(x)
        return x
    
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils import weight_norm
import math
from model.CVCNN import ComplexConv
import einops
class ReplicationPad1d(nn.Module):
    def __init__(self, padding) -> None:
        super(ReplicationPad1d, self).__init__()
        self.padding = padding

    def forward(self, input: Tensor) -> Tensor:
        replicate_padding = input[:, :, -1].unsqueeze(-1).repeat(1, 1, self.padding[-1])
        output = torch.cat([input, replicate_padding], dim=-1)
        return output

class high_freq_extract(nn.Module):
    def __init__(self,in_channels=128,out_channels=768):
        super(high_freq_extract, self).__init__()
        self.conv1 = ComplexConv(in_channels=in_channels, out_channels=out_channels//8, kernel_size=3, padding=1)
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_channels//4)


        self.conv2 = ComplexConv(in_channels=out_channels//8, out_channels=out_channels//4, kernel_size=3, padding=1)
        self.batchnorm2 = nn.BatchNorm1d(num_features=out_channels//2,)


        self.conv3 = ComplexConv(in_channels=out_channels//4, out_channels=out_channels//2, kernel_size=3, padding=1)
        self.batchnorm3 = nn.BatchNorm1d(num_features=out_channels)

        self.convres = ComplexConv(in_channels=in_channels, out_channels=out_channels // 2, kernel_size=3, padding=1)
        self.batchnormres = nn.BatchNorm1d(num_features=out_channels)


    def forward(self, sgn):
        x = self.conv1(sgn)
        x = F.relu(x)
        x = self.batchnorm1(x)


        x = self.conv2(x)
        x = F.relu(x)
        x = self.batchnorm2(x)


        x = self.conv3(x)
        x = F.relu(x)
        x = self.batchnorm3(x)

        res = self.convres(sgn)
        res = F.relu(res)
        res = self.batchnormres(res)

        x = x + res

        return x
    
class High_freq_conv_3layer(nn.Module):
    def __init__(self, d_model,out_channel, patch_len, stride, dropout):
        super(High_freq_conv_3layer, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = ReplicationPad1d((0, stride))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding =nn.Conv2d(1, d_model, kernel_size=(1, patch_len), stride=(1,stride), padding=(0, stride), bias=False)

        self.high_freq_exter=high_freq_extract(in_channels=d_model,out_channels=d_model*2)
        self.high_freq_exter2 = high_freq_extract(in_channels=d_model, out_channels=out_channel * 2)
        self.high_freq_exter3 = high_freq_extract(in_channels=out_channel, out_channels=out_channel * 2)
        self.pool=nn.MaxPool1d(kernel_size=2)
        self.drop=nn.Dropout(dropout)

    def forward(self, x):
        x = self.padding_patch_layer(x)
        x=torch.unsqueeze(x,1)
        x = self.value_embedding(x)
        x = self.drop(x)
        b,n,d,l=x.shape
        x = einops.rearrange(x, 'b d n l -> b (n d) l')
        
        x=self.high_freq_exter(x)
        x=self.pool(x)
        x = self.high_freq_exter2(x)
        x = self.pool(x)
        x = self.high_freq_exter3(x)
        x = self.pool(x)
        return x
    
class High_freq_conv(nn.Module):
    def __init__(self, d_model,out_channel, patch_len, stride, dropout):
        super(High_freq_conv, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = ReplicationPad1d((0, stride))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding =nn.Conv2d(1, d_model, kernel_size=(1, patch_len), stride=(1,stride), padding=(0, stride), bias=False)

        self.high_freq_exter=high_freq_extract(in_channels=d_model,out_channels=d_model*2)
        self.high_freq_exter2 = high_freq_extract(in_channels=d_model, out_channels=out_channel * 2)
        self.pool=nn.MaxPool1d(kernel_size=2)
        self.drop=nn.Dropout(dropout)

    def forward(self, x):
        x = self.padding_patch_layer(x)
        x=torch.unsqueeze(x,1)
        x = self.value_embedding(x)
        x = self.drop(x)
        b,n,d,l=x.shape
        x = einops.rearrange(x, 'b d n l -> b (n d) l')
        x=self.high_freq_exter(x)
        x=self.pool(x)
        x = self.high_freq_exter2(x)
        x = self.pool(x)
        return x


