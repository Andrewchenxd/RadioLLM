
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import einops
import math
import numpy as np
import transformers
from peft import get_peft_model, LoraConfig, AdaLoraConfig
from peft.peft_model import PeftModel
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer, PreTrainedTokenizerFast
from model.utils import *
from model.embed import DataEmbedding,PatchEmbedding,PatchEmbedding_hf
from math import sqrt
from model.norm import Normalize


transformers.logging.set_verbosity_error()
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        # x = self.dropout(x)
        return x
    
def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
    # Efficient implementation equivalent to the following:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value

class CrossAttention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
            var_num=None,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        if var_num is not None:
            self.template = nn.Parameter(
                torch.zeros(var_num, dim), requires_grad=True)
            torch.nn.init.normal_(self.template, std=.02)
        self.var_num = var_num

    def forward(self, x, query=None):
        B, N, C = x.shape
        if query is not None:
            q = self.q(query).reshape(
                B, query.shape[1], self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            q = self.q_norm(q)
            var_num = query.shape[1]
        else:
            q = self.q(self.template).reshape(1, self.var_num,
                                              self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            q = self.q_norm(q)
            q = q.repeat(B, 1, 1, 1)
            var_num = self.var_num
        kv = self.kv(x).reshape(B, N, 2, self.num_heads,
                                self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        k = self.k_norm(k)

        x = scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.,
        )

        x = x.transpose(1, 2).reshape(B, var_num, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Prompt_CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_norm=False, attn_drop=0., proj_drop=0., K=10):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.K = K

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.qk_norm = qk_norm
        # self.value_projection = nn.Linear(dim, d_keys * num_heads)

    def forward(self, prompt, enc_emd, word_emd):
        B, LP, D = prompt.shape
        B, LS, D = enc_emd.shape
        N, D = word_emd.shape

        # Compute query, key, and value
        qkv = self.qkv(torch.cat([prompt, enc_emd], dim=1))
        q, k, v = qkv.reshape(B, LP + LS, 3, self.num_heads, D // self.num_heads).permute(2, 0, 3, 1, 4)

        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if self.qk_norm:
            attn = attn / torch.sqrt(q.shape[-1])
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Compute attended values
        x = (attn @ v).transpose(1, 2).reshape(B, LP + LS, D)

        # Project attended values
        x = self.proj(x)
        x = self.proj_drop(x)
        # x_norm = F.normalize(x, dim=-1)
        # word_emd_norm = F.normalize(word_emd, dim=-1)
        # Find top-k word em
        sim = torch.einsum('bld,vd->bv', x, word_emd)

        # Find top-k values and indices
        topk_values, topk_indices = torch.topk(sim, k=self.K, dim=-1)

        # Gather top-k word embeddings
        topk_indices = topk_indices.unsqueeze(-1).expand(-1, -1, D)
        # 从 word_emd 中收集 top-k 词嵌入向量
        output = torch.gather(word_emd.expand(B, -1, -1), 1, topk_indices)
        return output

class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding

def soft_hard_prompt_topk(prompt_embeddings,prompt_word_embeddings,K):
    if K==0:
        return prompt_embeddings
    
    B, LP, D = prompt_embeddings.shape

    sim = torch.einsum('bld,vd->bv', prompt_embeddings, prompt_word_embeddings)

    # Find top-k values and indices
    topk_values, topk_indices = torch.topk(sim, k=K, dim=-1)

    # Gather top-k word embeddings
    topk_indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, D)
    # 从 word_emd 中收集 top-k 词嵌入向量
    prompt_embeddings = torch.gather(prompt_word_embeddings.expand(B, -1, -1), 1, topk_indices_expanded)
    return prompt_embeddings, topk_indices

def soft_hard_prompt(prompt_embeddings,prompt_word_embeddings,K):
    if K==0:
        return prompt_embeddings
    
    B, LP, D = prompt_embeddings.shape

    sim = torch.einsum('bld,vd->bv', prompt_embeddings, prompt_word_embeddings)

    # Find top-k values and indices
    topk_values, topk_indices = torch.topk(sim, k=K, dim=-1)

    # Gather top-k word embeddings
    topk_indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, D)
    # 从 word_emd 中收集 top-k 词嵌入向量
    prompt_embeddings = torch.gather(prompt_word_embeddings.expand(B, -1, -1), 1, topk_indices_expanded)
    return prompt_embeddings

class AttentionFusion(nn.Module):
    def __init__(self, in_channels):
        super(AttentionFusion, self).__init__()
        self.in_channels=in_channels
        self.query = nn.Linear(in_channels, in_channels)
        self.key = nn.Linear(in_channels, in_channels)
        self.value = nn.Linear(in_channels, in_channels)

    def forward(self, enc_out, enc_out_sgn):
        query = self.query(enc_out)
        key = self.key(enc_out_sgn)
        value = self.value(enc_out_sgn)

        attention_scores = torch.matmul(query, key.transpose(-2, -1))
        attention_scores = attention_scores / (self.in_channels ** 0.5)
        attention_weights = nn.functional.softmax(attention_scores, dim=-1)

        attended_features = torch.matmul(attention_weights, value)
        fused_features = enc_out + attended_features
        return fused_features
         
    
class RadioLLM(nn.Module):

    def __init__(self, configs, yaml_config):
        super(RadioLLM, self).__init__()
        self.task_name = configs.task_name

        self.d_ff = configs.d_ff
        self.top_k = 5
        self.d_llm = configs.llm_dim
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.is_LORA = configs.is_LORA
        self.decoder_is=configs.decoder_is
        self.load_lora_only=configs.load_lora_only

        if self.is_LORA:
            if configs.llm_model == 'GPT2':
                peft_config = LoraConfig(
                    r=8,  # LoRA矩阵的秩
                    lora_alpha=8,  # LoRA的缩放参数
                    lora_dropout=0.1,  # LoRA层的dropout率
                    target_modules=["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"],  # 要应用LoRA的模块名称
                )
            elif configs.llm_model == 'LLAMA3.2':
                peft_config = LoraConfig(
                    r=8,
                    lora_alpha=8,
                    target_modules=[
                        "self_attn.q_proj",
                        "self_attn.k_proj",
                        "self_attn.v_proj",
                        "self_attn.out_proj",
                        "mlp.fc1",
                        "mlp.fc2"
                    ],
                    lora_dropout=0.2,
                    bias="none",
                )
            elif configs.llm_model == 'BERT':
                peft_config = LoraConfig(
                    r=8,  # LoRA矩阵的秩
                    lora_alpha=8,  # LoRA的缩放参数
                    lora_dropout=0.1,  # LoRA层的dropout率
                    target_modules=["attention.self.query", "attention.self.key", "attention.self.value", "attention.output.dense", "intermediate.dense", "output.dense"]
                )
            else:
                raise ValueError("LoRA model not supported")

        self.cls_tokens = nn.Parameter(torch.zeros(
            1, 1, self.d_llm))
        torch.nn.init.normal_(self.cls_tokens, std=.02)
        self.mask_tokens = nn.ParameterDict({})
        self.right_prob = configs.right_prob
        self.mask_ratio = {}
        self.pred_len = {}
        self.seq_len = {}
        self.head_nf = {}
        self.description = {}
        # self.output_projection = nn.Sequential(
        #     nn.Linear(self.d_llm, self.d_llm // 2, bias=True),
        #     nn.Dropout(0.2),
        #     nn.Linear(self.d_llm // 2, configs.c_out, bias=True))
        self.output_projection = nn.ParameterDict({})
        self.predictor_ins = nn.ParameterDict({})
        self.predictor_cls = nn.ParameterDict({})
        self.nmb_prototypes = {}
        for i in range(len(yaml_config)):
            dataset_name = list(yaml_config.items())[i][1]['dataset_name']
            self.description[dataset_name] = list(yaml_config.items())[i][1]['content']
            self.min_mask_ratio = list(yaml_config.items())[i][1]['min_mask_ratio']
            self.max_mask_ratio = list(yaml_config.items())[i][1]['max_mask_ratio']
            self.mask_ratio[dataset_name] = [self.min_mask_ratio, self.max_mask_ratio]
            self.seq_len[dataset_name] = list(yaml_config.items())[i][1]['seq_len']
            self.pred_len[dataset_name] = self.seq_len[dataset_name]
            self.mask_tokens[dataset_name] = torch.zeros(1, 1, self.seq_len[dataset_name], 1)
            nn.init.normal_(self.mask_tokens[dataset_name], std=.02)
            if self.task_name in ['long_term_forecast', 'tsne', 'without_prompt', 'soft_hard_prompt',
                                  'soft_hard_prompt2', 'soft_hard_prompt3', 'classification']:
                self.patch_nums = int((self.seq_len[dataset_name] - self.patch_len) / self.stride + 2)
                self.head_nf[dataset_name] = self.d_ff * self.patch_nums
                self.output_projection[dataset_name] = FlattenHead(configs.enc_in, self.head_nf[dataset_name],
                                                                   self.pred_len[dataset_name],
                                                                   head_dropout=configs.dropout)
                self.nmb_prototypes[dataset_name] = list(yaml_config.items())[i][1]['numclass']
                self.predictor_ins[dataset_name] = nn.Sequential(
                    nn.Linear((self.d_llm - self.d_ff) * 2, (self.d_llm - self.d_ff) * 2),
                    nn.ReLU(),
                    nn.Linear((self.d_llm - self.d_ff) * 2, 256)
                    
                )

        # self.category_tokens = nn.Parameter(torch.zeros(
        #     1, 1, self.d_llm))
        # torch.nn.init.normal_(self.category_tokens, std=.02)
        self.llm_model_name = configs.llm_model
        if configs.llm_model == 'LLAMA':
            # self.llama_config = LlamaConfig.from_pretrained('/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/')
            self.llama_config = LlamaConfig.from_pretrained(configs.llm_path)
            self.llama_config.num_hidden_layers = configs.llm_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            try:
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    configs.llm_path,
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    configs.llm_path,
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            try:
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    configs.llm_path,
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'LLAMA3.2':
            self.llama_config = LlamaConfig.from_pretrained(configs.llm_path)
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            try:
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    configs.llm_path,
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    configs.llm_path,
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            try:
                self.tokenizer = PreTrainedTokenizerFast.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    configs.llm_path,
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = PreTrainedTokenizerFast.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'GPT2':
            self.gpt2_config = GPT2Config.from_pretrained(configs.llm_path)

            self.gpt2_config.num_hidden_layers = configs.llm_layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            try:
                self.llm_model = GPT2Model.from_pretrained(
                    configs.llm_path,
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.gpt2_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = GPT2Model.from_pretrained(
                    configs.llm_path,
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.gpt2_config,
                )

            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    configs.llm_path,
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    configs.llm_path,
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'BERT':
            self.bert_config = BertConfig.from_pretrained(configs.llm_path)
            self.bert_config.num_hidden_layers = configs.llm_layers
            self.bert_config.output_attentions = True
            self.bert_config.output_hidden_states = True
            try:
                self.llm_model = BertModel.from_pretrained(
                    configs.llm_path,
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.bert_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.bert_config,
                )
            try:
                self.tokenizer = BertTokenizer.from_pretrained(
                    configs.llm_path,
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'GPT2_w/o_pretrain':
            self.gpt2_config = GPT2Config.from_pretrained(configs.llm_path)

            self.gpt2_config.num_hidden_layers = configs.llm_layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            self.llm_model = GPT2Model(config=self.gpt2_config)

            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    configs.llm_path,
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    configs.llm_path,
                    trust_remote_code=True,
                    local_files_only=False
                )
        else:
            raise Exception('LLM model is not defined')

        print(self.llm_model)
        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        self.high_freq_extract = High_freq_conv_3layer(d_model=configs.d_model,
                                                out_channel=self.d_llm, patch_len=self.patch_len//4,
                                                stride=1, dropout=0)
        # self.learn_able_a = nn.Parameter(torch.randn((self.d_llm - self.d_ff) * 2))
        # self.learn_able_b = nn.Parameter(torch.randn((self.d_llm - self.d_ff) * 2))
        self.attn_fusion = AttentionFusion(self.d_llm)
        self.dropout = nn.Dropout(configs.dropout)
        self.dec_embedding = DataEmbedding(configs.enc_in, self.d_llm, configs.dropout)
        Attn = ProbAttention
        if self.task_name not in ['classificationwithout_prompt']:
            self.decoder = Decoder(
                [
                    DecoderLayer(
                        AttentionLayer(Attn(configs.decode_mask, configs.factor, attention_dropout=configs.dropout,
                                            output_attention=False),
                                       self.d_llm, configs.n_heads, mix=configs.mix, attn=configs.attn),
                        AttentionLayer(
                            FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                          output_attention=False),
                            self.d_llm, configs.n_heads, mix=False),
                        self.d_llm,
                        configs.d_ff2,
                        dropout=configs.dropout,
                        activation=configs.activation,
                    )
                    for l in range(configs.d_layers)
                ],
                norm_layer=torch.nn.LayerNorm(self.d_llm)
            )

        self.patch_embedding = PatchEmbedding(
            configs.d_model, self.patch_len, self.stride, configs.dropout)

        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

        self.reprogramming_layer = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm, attention_dropout=configs.dropout)

        self.normalize_layers = Normalize(configs.enc_in, affine=False)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.K = configs.K
        if self.task_name in ['classification', 'classificationwithout_prompt']:
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.proj_in = nn.Linear(self.d_llm, self.d_llm // 2)
            self.cross_att = CrossAttention(self.d_llm // 2)

            # self.mlp_head = nn.Sequential(nn.Linear(self.d_llm*2, 128),
            #                               nn.ReLU(),
            #                               nn.Dropout(0.1),
            #                               nn.Linear(128, configs.numclass))
            self.mlp_head = nn.Sequential(nn.Linear(self.d_llm*2, 512),
                                          nn.ReLU(),
                                          nn.Dropout(0.1),
                                          nn.Linear(512, configs.numclass))
            self.fc_head = ResNet1d_enc(numclass=configs.numclass)
        elif self.task_name in ['classificationsnr']: 
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.proj_in = nn.Linear(self.d_llm, self.d_llm // 2)
            self.cross_att = CrossAttention(self.d_llm // 2)

            # self.mlp_head = nn.Sequential(nn.Linear(self.d_llm*2, 128),
            #                               nn.ReLU(),
            #                               nn.Dropout(0.1),
            #                               nn.Linear(128, configs.numclass))
            self.snr_head = nn.Sequential(nn.Linear(self.d_llm*2, 128),
                                          nn.ReLU(),
                                          nn.Dropout(0.1),
                                          nn.Linear(128, 1))
            
        elif self.task_name in ['soft_hard_prompt']:
            self.mapping_layer2 = nn.Linear(self.vocab_size, self.num_tokens)
            self.prompt_attn = Prompt_CrossAttention(self.d_llm, num_heads=configs.n_heads, qkv_bias=False,
                                                     qk_norm=False, attn_drop=configs.dropout,
                                                     proj_drop=configs.dropout, K=self.K)
        elif self.task_name in ['soft_hard_prompt2']:
            self.mapping_layer2 = nn.Linear(self.vocab_size, self.num_tokens)

        if self.is_LORA == False:
            for param in self.llm_model.parameters():
                param.requires_grad = False
        else:
            self.llm_model = get_peft_model(self.llm_model, peft_config)
            for name, param in self.llm_model.named_parameters():
                if 'lora' not in name:  # Assuming LoRA parameters have 'lora' in their names
                    param.requires_grad = False
                elif self.load_lora_only and 'lora' in name:
                    param.requires_grad = False

        # 初始化prompt_embeddings_log
        self.prompt_embeddings_log = []



    def random_masking(self, x, min_mask_ratio, max_mask_ratio):
        """
        Perform per-sample random masking.
        """
        N, V, L, D = x.shape  # batch, var, length, dim

        # Calculate mask ratios and lengths to keep for each sample in the batch
        mask_ratios = torch.rand(N, device=x.device) * \
                      (max_mask_ratio - min_mask_ratio) + min_mask_ratio
        len_keeps = (L * (1 - mask_ratios)).long()

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        # ascend: small is keep, large is remove
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)

        # Create a range tensor and compare with len_keeps for mask generation
        range_tensor = torch.arange(L, device=x.device).expand(N, L)
        mask = (range_tensor >= len_keeps.unsqueeze(1))

        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        mask = mask.float()

        return mask

    def continuous_masking(self, x, min_mask_ratio, max_mask_ratio):
        """
        Perform per-sample continuous masking.
        """
        min_mask_ratio, max_mask_ratio = min_mask_ratio / 2, max_mask_ratio / 2
        N, V, L, D = x.shape  # batch, var, length, dim

        # Initialize mask
        mask = torch.zeros([N, L], device=x.device)

        for i in range(N):
            # Determine the length of the masked region for this sample
            mask_length = int(torch.randint(int(min_mask_ratio * L), int(max_mask_ratio * L) + 1, (1,)).item())

            # Randomly choose the start point of the masked region
            start = torch.randint(0, L - mask_length + 1, (1,)).item()

            # Apply mask
            mask[i, start:start + mask_length] = 1

        return mask.float()

    def right_masking(self, x, min_mask_ratio, max_mask_ratio):
        N, V, L, D = x.shape  # batch, var, length, dim

        # Randomly choose a mask ratio for each sample within the specified range
        mask_ratios = torch.rand(N, device=x.device) * \
                      (max_mask_ratio - min_mask_ratio) + min_mask_ratio
        len_keeps = (L * (1 - mask_ratios)).long()

        # Binary mask creation without a for loop
        len_keeps_matrix = len_keeps.unsqueeze(1).expand(N, L)
        indices = torch.arange(L, device=x.device).expand_as(len_keeps_matrix)
        mask = indices >= len_keeps_matrix
        mask = mask.float()

        return mask

    def choose_masking(self, x, right_prob, min_mask_ratio, max_mask_ratio):
        # Generate a random number to decide which masking function to use
        if torch.rand(1).item() > right_prob:
            return self.random_masking(x, min_mask_ratio, max_mask_ratio)
        else:
            return self.continuous_masking(x, min_mask_ratio, max_mask_ratio)

    def forward(self, x_enc, enable_mask=False, dataset_name=None):
        if self.task_name in ['long_term_forecast', 'soft_hard_prompt', 'soft_hard_prompt2', 'soft_hard_prompt3']:
            if enable_mask:
                dec_out, mask_seq = self.forecast(x_enc, enable_mask, dataset_name)

                return dec_out[:, -self.pred_len[dataset_name]:, :], mask_seq
            else:
                dec_out = self.forecast(x_enc, enable_mask, dataset_name)

                return dec_out[:, -self.pred_len[dataset_name]:, :]
        if self.task_name == 'without_prompt':
            dec_out = self.forecast_without_prompt(x_enc)
            return dec_out[:, -self.pred_len[dataset_name]:, :]
        if self.task_name == 'classification':
            out = self.classific(x_enc, enable_mask, dataset_name)
            return out
        if self.task_name == 'classificationsnr':
            out = self.classificsnr(x_enc, enable_mask, dataset_name)
            return out
        if self.task_name == 'classificationwithout_prompt':
            out = self.classific_without_prompt(x_enc)
            return out
        if self.task_name == 'classificationwithenc':
            out = self.classific_withenc(x_enc, enable_mask, dataset_name)
            return out
        if self.task_name == 'tsne':
            dec_out_q = self.tsne(x_enc, enable_mask, dataset_name)
            # dec_out_q = self.pool(dec_out_q.transpose(1, 2))
            # dec_out_q = dec_out_q.view(dec_out_q.size(0), -1)

            # dec_out_q= nn.functional.normalize(dec_out_q, dim=1)
            return dec_out_q
        if self.task_name =="embed":
            dec_out_q = self.embed(x_enc, enable_mask, dataset_name)
            # dec_out_q = self.pool(dec_out_q.transpose(1, 2))
            # dec_out_q = dec_out_q.view(dec_out_q.size(0), -1)

            # dec_out_q= nn.functional.normalize(dec_out_q, dim=1)
            return dec_out_q
        return None

    def forecast(self, x_enc, enable_mask=False, dataset_name=None, encoder_k_is=False):
        x_enc = x_enc.permute(0, 2, 1)

        x_enc = self.normalize_layers(x_enc, 'norm')

        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        min_values = torch.min(x_enc, dim=1)[0]
        max_values = torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values
        lags = self.calcute_lags(x_enc)
        trends = x_enc.diff(dim=1).sum(dim=1)

        prompt = []
        for b in range(x_enc.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            lags_values_str = str(lags[b].tolist())
            if enable_mask == False:
                task_prompt = f"Task description: denoising a radio signal based on {str(self.pred_len)} samples with noise; "
            else:
                task_prompt = f"Task description: recovering a missing radio signal based on {str(self.pred_len)} samples; "
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.description[dataset_name]}"
                f"{task_prompt}"
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                f"top 5 lags are : {lags_values_str}<|<end_prompt>|>"
            )

            prompt.append(prompt_)

        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()

        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))  # (batch, prompt_token, dim)

        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)

        x_enc = x_enc.permute(0, 2, 1).contiguous()

        if enable_mask:
            x_enc = x_enc.unsqueeze(dim=-1)
            mask = self.choose_masking(x_enc, self.right_prob,
                                       self.mask_ratio[dataset_name][0], self.mask_ratio[dataset_name][1])
            mask_repeat = mask.unsqueeze(dim=1).unsqueeze(dim=-1)
            mask_repeat = mask_repeat.repeat(1, x_enc.shape[1], 1, x_enc.shape[-1])
            x_enc = x_enc * (1 - mask_repeat) + self.mask_tokens[dataset_name] * mask_repeat
            mask_seq = mask
            x_enc = x_enc.squeeze(dim=-1)

        enc_out_sgn, n_vars = self.patch_embedding(x_enc)
        enc_out_sgn_1 = self.high_freq_extract(x_enc)
        enc_out_sgn_1 = einops.rearrange(enc_out_sgn_1, 'b (n d) l -> (b n) l d',n=n_vars)
        enc_out = self.reprogramming_layer(enc_out_sgn, source_embeddings, source_embeddings)
        enc_out = self.attn_fusion(enc_out, enc_out_sgn_1)
        # enc_out = enc_out_sgn_1
        if self.task_name in ['soft_hard_prompt']:
            prompt_word_embeddings = self.mapping_layer2(self.word_embeddings.permute(1, 0)).permute(1, 0)
            prompt_embeddings = self.prompt_attn(prompt_embeddings, enc_out, prompt_word_embeddings)
        elif self.task_name in ['soft_hard_prompt2']:
            prompt_word_embeddings = self.mapping_layer2(self.word_embeddings.permute(1, 0)).permute(1, 0)
            prompt_embeddings = soft_hard_prompt(prompt_embeddings, prompt_word_embeddings, self.K)
        elif self.task_name in ['soft_hard_prompt3']:
            prompt_word_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)
            prompt_embeddings = soft_hard_prompt(prompt_embeddings, prompt_word_embeddings, self.K)
        cls_tokens = self.cls_tokens.repeat(enc_out.shape[0], 1, 1)
        self.prompt_length = prompt_embeddings.shape[1]
        llama_enc_out = torch.cat([prompt_embeddings, enc_out, cls_tokens], dim=1)
        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        # dec_out = dec_out[:, :, :self.d_ff]
        dec_out = torch.reshape(
            dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()
        dec_out = dec_out[:, :, :, self.prompt_length:-1]  # B,var,dim,len
        if self.decoder_is:
            dec_out_1 = einops.rearrange(dec_out, 'b n d l -> b (n l) d')
            dec_out = self.decoder(dec_out_1, dec_out_1, x_mask=None, cross_mask=None)
            dec_out = einops.rearrange(dec_out, 'b (n l) d -> b n d l', n=2)
        dec_out = dec_out[:, :, :self.d_ff]
        dec_out = self.output_projection[dataset_name](dec_out)
        dec_out = dec_out.permute(0, 2, 1).contiguous()
        dec_out = self.normalize_layers(dec_out, 'denorm')
        dec_out = dec_out.permute(0, 2, 1)
        if enable_mask:
            return dec_out, mask_seq
        else:
            return dec_out

    def forecast_without_prompt(self, x_enc):
        x_enc = x_enc.permute(0, 2, 1)
        x_enc = self.normalize_layers(x_enc, 'norm')

        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()

        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)

        x_enc = x_enc.permute(0, 2, 1).contiguous()
        enc_out, n_vars = self.patch_embedding(x_enc)
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)

        cls_tokens = self.cls_tokens.repeat(enc_out.shape[0], 1, 1)

        llama_enc_out = torch.cat([enc_out, cls_tokens], dim=1)
        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        dec_out = dec_out[:, :, :self.d_ff]

        dec_out = torch.reshape(
            dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()

        dec_out = self.output_projection(dec_out[:, :, :, :-1])
        dec_out = dec_out.permute(0, 2, 1).contiguous()

        dec_out = self.normalize_layers(dec_out, 'denorm')
        dec_out = dec_out.permute(0, 2, 1)
        return dec_out

    def embed(self, x_enc, enable_mask=False, dataset_name=None, encoder_k_is=False):
        x_enc_ori=x_enc
        x_enc = x_enc.permute(0, 2, 1)

        x_enc = self.normalize_layers(x_enc, 'norm')

        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        min_values = torch.min(x_enc, dim=1)[0]
        max_values = torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values
        lags = self.calcute_lags(x_enc)
        trends = x_enc.diff(dim=1).sum(dim=1)

        prompt = []
        for b in range(x_enc.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            lags_values_str = str(lags[b].tolist())
            if enable_mask == False:
                task_prompt = f"Task description: denoising a radio signal based on {str(self.pred_len)} samples with noise; "
            else:
                task_prompt = f"Task description: recovering a missing radio signal based on {str(self.pred_len)} samples; "
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.description[dataset_name]}"
                f"{task_prompt}"
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                f"top 5 lags are : {lags_values_str}<|<end_prompt>|>"
            )

            prompt.append(prompt_)

        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()

        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))  # (batch, prompt_token, dim)

        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)

        x_enc = x_enc.permute(0, 2, 1).contiguous()

        if enable_mask:
            x_enc = x_enc.unsqueeze(dim=-1)
            mask = self.choose_masking(x_enc, self.right_prob,
                                       self.mask_ratio[dataset_name][0], self.mask_ratio[dataset_name][1])
            mask_repeat = mask.unsqueeze(dim=1).unsqueeze(dim=-1)
            mask_repeat = mask_repeat.repeat(1, x_enc.shape[1], 1, x_enc.shape[-1])
            x_enc = x_enc * (1 - mask_repeat) + self.mask_tokens[dataset_name] * mask_repeat
            mask_seq = mask
            x_enc = x_enc.squeeze(dim=-1)

        enc_out_sgn, n_vars = self.patch_embedding(x_enc)
        enc_out_sgn_1 = self.high_freq_extract(x_enc_ori)
        enc_out_sgn_1 = einops.rearrange(enc_out_sgn_1, 'b (n d) l -> (b n) l d', n=n_vars)

        enc_out = self.reprogramming_layer(enc_out_sgn, source_embeddings, source_embeddings)
        enc_out = self.attn_fusion(enc_out, enc_out_sgn_1)
        # enc_out = enc_out_sgn_1
        prompt_word_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)
        prompt_embeddings = soft_hard_prompt(prompt_embeddings, prompt_word_embeddings, self.K)
        cls_tokens = self.cls_tokens.repeat(enc_out.shape[0], 1, 1)
        self.prompt_length = prompt_embeddings.shape[1]
        llama_enc_out = torch.cat([prompt_embeddings, enc_out, cls_tokens], dim=1)
        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        dec_out = einops.rearrange(dec_out, '(b n) l d -> b l (n d)', n=2)
        dec_out_con= dec_out[:, self.prompt_length:-1, :]
        dec_out_con=self.pool(dec_out_con.transpose(1, 2))
        dec_out_con = dec_out_con.view(dec_out_con.size(0), -1)
        # dec_out_con = self.mlp_head(dec_out_con)
        return dec_out_con
    
    def classific(self, x_enc, enable_mask=False, dataset_name=None, encoder_k_is=False):
            
        x_enc_ori=x_enc
        x_enc = x_enc.permute(0, 2, 1)

        x_enc = self.normalize_layers(x_enc, 'norm')

        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        min_values = torch.min(x_enc, dim=1)[0]
        max_values = torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values
        lags = self.calcute_lags(x_enc)
        trends = x_enc.diff(dim=1).sum(dim=1)

        prompt = []
        for b in range(x_enc.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            lags_values_str = str(lags[b].tolist())
            if enable_mask == False:
                task_prompt = f"Task description: denoising a radio signal based on {str(self.pred_len)} samples with noise; "
            else:
                task_prompt = f"Task description: recovering a missing radio signal based on {str(self.pred_len)} samples; "
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.description[dataset_name]}"
                f"{task_prompt}"
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                f"top 5 lags are : {lags_values_str}<|<end_prompt>|>"
            )

            prompt.append(prompt_)

        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()

        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))  # (batch, prompt_token, dim)

        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)

        x_enc = x_enc.permute(0, 2, 1).contiguous()

        if enable_mask:
            x_enc = x_enc.unsqueeze(dim=-1)
            mask = self.choose_masking(x_enc, self.right_prob,
                                       self.mask_ratio[dataset_name][0], self.mask_ratio[dataset_name][1])
            mask_repeat = mask.unsqueeze(dim=1).unsqueeze(dim=-1)
            mask_repeat = mask_repeat.repeat(1, x_enc.shape[1], 1, x_enc.shape[-1])
            x_enc = x_enc * (1 - mask_repeat) + self.mask_tokens[dataset_name] * mask_repeat
            mask_seq = mask
            x_enc = x_enc.squeeze(dim=-1)

        enc_out_sgn, n_vars = self.patch_embedding(x_enc)
        enc_out_sgn_1 = self.high_freq_extract(x_enc_ori)
        enc_out_sgn_1 = einops.rearrange(enc_out_sgn_1, 'b (n d) l -> (b n) l d', n=n_vars)

        enc_out = self.reprogramming_layer(enc_out_sgn, source_embeddings, source_embeddings)
        enc_out = self.attn_fusion(enc_out, enc_out_sgn_1)
        # enc_out = enc_out_sgn_1
        prompt_word_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)
        prompt_embeddings = soft_hard_prompt(prompt_embeddings, prompt_word_embeddings, self.K)

        cls_tokens = self.cls_tokens.repeat(enc_out.shape[0], 1, 1)
        self.prompt_length = prompt_embeddings.shape[1]
        llama_enc_out = torch.cat([prompt_embeddings, enc_out, cls_tokens], dim=1)
        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        dec_out = einops.rearrange(dec_out, '(b n) l d -> b l (n d)', n=2)
        dec_out_con= dec_out[:, self.prompt_length:-1, :]
        dec_out_con=self.pool(dec_out_con.transpose(1, 2))
        dec_out_con = dec_out_con.view(dec_out_con.size(0), -1)
        dec_out_con = self.mlp_head(dec_out_con)
        return dec_out_con


    def classificsnr(self, x_enc, enable_mask=False, dataset_name=None, encoder_k_is=False):
            
        x_enc_ori=x_enc
        x_enc = x_enc.permute(0, 2, 1)

        x_enc = self.normalize_layers(x_enc, 'norm')

        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        min_values = torch.min(x_enc, dim=1)[0]
        max_values = torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values
        lags = self.calcute_lags(x_enc)
        trends = x_enc.diff(dim=1).sum(dim=1)

        prompt = []
        for b in range(x_enc.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            lags_values_str = str(lags[b].tolist())
            if enable_mask == False:
                task_prompt = f"Task description: denoising a radio signal based on {str(self.pred_len)} samples with noise; "
            else:
                task_prompt = f"Task description: recovering a missing radio signal based on {str(self.pred_len)} samples; "
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.description[dataset_name]}"
                f"{task_prompt}"
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                f"top 5 lags are : {lags_values_str}<|<end_prompt>|>"
            )

            prompt.append(prompt_)

        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()

        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))  # (batch, prompt_token, dim)

        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)

        x_enc = x_enc.permute(0, 2, 1).contiguous()

        if enable_mask:
            x_enc = x_enc.unsqueeze(dim=-1)
            mask = self.choose_masking(x_enc, self.right_prob, self.mask_ratio[dataset_name][0], self.mask_ratio[dataset_name][1])
            mask_repeat = mask.unsqueeze(dim=1).unsqueeze(dim=-1)
            mask_repeat = mask_repeat.repeat(1, x_enc.shape[1], 1, x_enc.shape[-1])
            x_enc = x_enc * (1 - mask_repeat) + self.mask_tokens[dataset_name] * mask_repeat
            mask_seq = mask
            x_enc = x_enc.squeeze(dim=-1)

        enc_out_sgn, n_vars = self.patch_embedding(x_enc)
        enc_out_sgn_1 = self.high_freq_extract(x_enc_ori)
        enc_out_sgn_1 = einops.rearrange(enc_out_sgn_1, 'b (n d) l -> (b n) l d', n=n_vars)
        enc_out = self.reprogramming_layer(enc_out_sgn, source_embeddings, source_embeddings)
        enc_out = self.attn_fusion(enc_out, enc_out_sgn_1)
        prompt_word_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)
        prompt_embeddings = soft_hard_prompt(prompt_embeddings, prompt_word_embeddings, self.K)
        cls_tokens = self.cls_tokens.repeat(enc_out.shape[0], 1, 1)
        self.prompt_length = prompt_embeddings.shape[1]
        llama_enc_out = torch.cat([prompt_embeddings, enc_out, cls_tokens], dim=1)
        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        dec_out = einops.rearrange(dec_out, '(b n) l d -> b l (n d)', n=2)
        dec_out_con= dec_out[:, self.prompt_length:-1, :]
        dec_out_con=self.pool(dec_out_con.transpose(1, 2))
        dec_out_con = dec_out_con.view(dec_out_con.size(0), -1)
        dec_out_con = self.snr_head(dec_out_con)
        return dec_out_con
    
    def classific_withenc(self, x_enc, enable_mask=False, dataset_name=None):
        x_enc = x_enc.permute(0, 2, 1)
        x_enc = self.normalize_layers(x_enc, 'norm')

        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        min_values = torch.min(x_enc, dim=1)[0]
        max_values = torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values
        lags = self.calcute_lags(x_enc)
        trends = x_enc.diff(dim=1).sum(dim=1)

        prompt = []
        for b in range(x_enc.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            lags_values_str = str(lags[b].tolist())
            if enable_mask == False:
                task_prompt = f"Task description: denoising a radio signal based on {str(self.pred_len)} samples with noise; "
            else:
                task_prompt = f"Task description: recovering a missing radio signal based on {str(self.pred_len)} samples; "
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.description[dataset_name]}"
                f"{task_prompt}"
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                f"top 5 lags are : {lags_values_str}<|<end_prompt>|>"
            )

            prompt.append(prompt_)

        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()

        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))  # (batch, prompt_token, dim)

        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)

        x_enc = x_enc.permute(0, 2, 1).contiguous()
        if enable_mask:
            x_enc = x_enc.unsqueeze(dim=-1)
            mask = self.choose_masking(x_enc, self.right_prob,
                                       self.mask_ratio[dataset_name][0], self.mask_ratio[dataset_name][1])
            mask_repeat = mask.unsqueeze(dim=1).unsqueeze(dim=-1)
            mask_repeat = mask_repeat.repeat(1, x_enc.shape[1], 1, x_enc.shape[-1])
            x_enc = x_enc * (1 - mask_repeat) + self.mask_tokens[dataset_name] * mask_repeat
            mask_seq = mask
            x_enc = x_enc.squeeze(dim=-1)
        enc_out, n_vars = self.patch_embedding(x_enc)

        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)

        prompt_word_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)
        prompt_embeddings = soft_hard_prompt(prompt_embeddings, prompt_word_embeddings, self.K)
        cls_tokens = self.cls_tokens.repeat(enc_out.shape[0], 1, 1)
        self.prompt_length = prompt_embeddings.shape[1]
        enc_out = torch.cat([prompt_embeddings, enc_out, cls_tokens], dim=1)
        enc_out = torch.reshape(
            enc_out, (-1, n_vars * enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.transpose(1, 2)
        if enable_mask:
            return enc_out, cls_tokens
        else:
            return enc_out

    def classific_without_prompt(self, x_enc):
        x_enc = x_enc.permute(0, 2, 1)
        x_enc = self.normalize_layers(x_enc, 'norm')

        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()

        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)

        x_enc = x_enc.permute(0, 2, 1).contiguous()
        enc_out, n_vars = self.patch_embedding(x_enc)
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)

        cls_tokens = self.cls_tokens.repeat(enc_out.shape[0], 1, 1)

        llama_enc_out = torch.cat([enc_out, cls_tokens], dim=1)

        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        dec_out = torch.reshape(
            dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        # dec_out = einops.rearrange(dec_out, 'b n l c -> b (n l) c')
        # dec_out= dec_out.permute(0, 2, 1)
        # dec_out =self.pool(dec_out)
        # latent = torch.flatten(dec_out, 1)
        # out = self.mlp_head(latent)

        x = self.proj_in(dec_out)
        B, V, L, C = x.shape
        x = x.view(-1, L, C)
        cls_token = x[:, -1:]
        cls_token = self.cross_att(x, query=cls_token)
        cls_token = cls_token.reshape(B, V, -1, C)
        cls_token = einops.rearrange(cls_token, 'b n l c -> b (n l) c')
        cls_token = cls_token.permute(0, 2, 1)
        cls_token = self.pool(cls_token)
        cls_token = torch.flatten(cls_token, 1)
        out = self.mlp_head(cls_token)
        return out

    def calcute_lags(self, x_enc):
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags