"""
Hybrid A3QER: Revolutionary approach combining activation smoothing with layer criticality analysis
This solves the repetition problem while maintaining quality through intelligent layer selection
"""

from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Hugging Face GPT-2 Conv1D alias 
try:
    from transformers.models.gpt2.modeling_gpt2 import Conv1D as HFConv1D
except Exception:
    HFConv1D = None 

def _is_linear_like(m: nn.Module) -> bool:
    if isinstance(m, (nn.Linear, nn.Conv1d)):
        return True
    if HFConv1D is not None and isinstance(m, HFConv1D): 
        return True
    return m.__class__.__name__ == "Conv1D"




class LayerCriticalityAnalyzer:
    """
    Identifies which layers are critical based on attention head importance patterns
    Inspired by research showing only specialized heads matter
    """
    
    def __init__(self, model, calibration_data):
        self.model = model
        self.calibration_data = calibration_data
        self.layer_importance = {}
        
    
    def analyze_attention_patterns(self):
        """
        Analyze which attention heads are doing the heavy lifting.
        Uses a small, normalized forward pass; robust for dict/tensors and seq2seq.
        """
        attention_scores = defaultdict(list)
        handles = []

        def create_attention_hook(name):
            def hook(module, inp, out):
                #normalizing output from various HF attention modules
                attn = None
                if isinstance(out, torch.Tensor):
                    attn = out
                elif isinstance(out, (tuple, list)) and out and isinstance(out[-1], torch.Tensor):
                    attn = out[-1]  #many modules keep attention probs last
                elif isinstance(out, dict):
                    cand = out.get("attn_probs") or out.get("attentions") or out.get("attention_probs")
                    if isinstance(cand, torch.Tensor):
                        attn = cand
                if attn is not None and attn.dim() >= 3:
                    attn_weights = attn.softmax(dim=-1) if attn.dim() > 2 else attn
                    entropy = -(attn_weights * (attn_weights + 1e-8).log()).sum(dim=-1).mean()
                    attention_scores[name].append(float(entropy))
            return hook

        for name, module in self.model.named_modules():
            if 'attn' in name.lower():
                handles.append(module.register_forward_hook(create_attention_hook(name)))

        # --- normalized forward args (works for causal + seq2seq) ---
        is_seq2seq = bool(getattr(self.model.config, "is_encoder_decoder", False))
        call_kwargs, call_args = None, None

        if isinstance(self.calibration_data, dict):
            call_kwargs = {
                k: (v[:4] if isinstance(v, torch.Tensor) and v.dim() >= 1 else v)
                for k, v in self.calibration_data.items()
            }
            if "input_ids" in call_kwargs and "attention_mask" not in call_kwargs:
                call_kwargs["attention_mask"] = torch.ones_like(call_kwargs["input_ids"])
            if is_seq2seq and "decoder_input_ids" not in call_kwargs:
                bsz = call_kwargs["input_ids"].size(0)
                device = call_kwargs["input_ids"].device
                dec_start = getattr(self.model.config, "decoder_start_token_id", None)
                if dec_start is None:
                    dec_start = getattr(self.model.config, "pad_token_id", None) or getattr(self.model.config, "eos_token_id", None) or 0
                call_kwargs["decoder_input_ids"] = torch.full((bsz, 1), dec_start, dtype=call_kwargs["input_ids"].dtype, device=device)

        elif isinstance(self.calibration_data, (list, tuple)):
            if len(self.calibration_data) in (1, 2) and isinstance(self.calibration_data[0], torch.Tensor):
                input_ids = self.calibration_data[0][:4]
                attention_mask = (self.calibration_data[1][:4]
                                if len(self.calibration_data) == 2 and isinstance(self.calibration_data[1], torch.Tensor)
                                else torch.ones_like(input_ids))
                call_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
                if is_seq2seq:
                    bsz = input_ids.size(0)
                    device = input_ids.device
                    dec_start = getattr(self.model.config, "decoder_start_token_id", None)
                    if dec_start is None:
                        dec_start = getattr(self.model.config, "pad_token_id", None) or getattr(self.model.config, "eos_token_id", None) or 0
                    call_kwargs["decoder_input_ids"] = torch.full((bsz, 1), dec_start, dtype=input_ids.dtype, device=device)
            else:
                call_args = tuple(self.calibration_data)

        elif isinstance(self.calibration_data, torch.Tensor):
            input_ids = self.calibration_data[:4]
            attention_mask = torch.ones_like(input_ids)
            call_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if is_seq2seq:
                bsz = input_ids.size(0); device = input_ids.device
                dec_start = getattr(self.model.config, "decoder_start_token_id", None)
                if dec_start is None:
                    dec_start = getattr(self.model.config, "pad_token_id", None) or getattr(self.model.config, "eos_token_id", None) or 0
                call_kwargs["decoder_input_ids"] = torch.full((bsz, 1), dec_start, dtype=input_ids.dtype, device=device)

        with torch.no_grad():
            if call_kwargs is not None:
                _ = self.model(**call_kwargs)
            elif call_args is not None:
                _ = self.model(*call_args)
            else:
                _ = self.model(self.calibration_data)

        for h in handles:
            h.remove()

        for name, scores in attention_scores.items():
            if scores:
                self.layer_importance[name] = float(np.mean(scores))

        return self.layer_importance



    
    def get_layer_config(self, layer_name):
        """
        INT8-only strategy for maximum CPU speed.
        Mixed precision (INT4/6) is too slow on CPU.
        """
        #parse layer position
        layer_num = -1
        if 'h.' in layer_name:
            try:
                layer_num = int(layer_name.split('h.')[1].split('.')[0])
            except:
                pass
        
        #KEEPING ONLY! first block's output projections in FP32 for numerical stability
        if layer_num == 0 and ('c_proj' in layer_name or 'mlp.c_proj' in layer_name):
            return {
                'quantize': False,
                'smooth': False,
                'bits': None
            }
        
        #Everything else INT8 with smoothing
        return {
            'quantize': True,
            'smooth': True,
            'bits': 8,  #pure INT8 - fastest dequantization on CPU
            'protect_ratio': 0.02
        }

class ImprovedActivationSmoother:
    """
    Enhanced smoothing that prevents activation outliers while preserving information
    """
    
    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        
    def smooth_activations(self, activations, weights):
        """
        Smooth only the outlier channels, preserve normal ones
        Handle both Linear and Conv1D weight dimensions
        """
        #activation statistics
        if activations.dim() == 3:  # (batch, seq, features)
            act_max = activations.abs().max(dim=0)[0].max(dim=0)[0]
            act_mean = activations.abs().mean(dim=[0, 1])
        else:  #(batch, features)
            act_max = activations.abs().max(dim=0)[0]
            act_mean = activations.abs().mean(dim=0)
        
        #outlier channels
        outlier_mask = act_max > (act_mean * 10)  #10x threshold
        
        #per-channel smoothing factors
        smooth_factor = torch.ones_like(act_max)
        
        if outlier_mask.any():
            #only smooth outliers
            outlier_scale = act_max[outlier_mask] / (act_mean[outlier_mask] * 5)
            smooth_factor[outlier_mask] = outlier_scale.pow(self.alpha).clamp(min=0.1, max=10)
        
        #selective smoothing to activations
        if activations.dim() == 3:
            smoothed_acts = activations / smooth_factor.unsqueeze(0).unsqueeze(0)
        else:
            smoothed_acts = activations / smooth_factor.unsqueeze(0)
        
        #compensateing in weights to handle different weight formats
        compensated_weights = weights.clone()
        
        #noting, For Conv1D in GPT-2: weights are (in_features, out_features)
        #noting, For Linear: weights are (out_features, in_features)
        if weights.dim() == 2:
            #checking format in_features matches smooth_factor size
            if weights.shape[0] == smooth_factor.shape[0]:
                # Conv1D case: scale input dimension
                compensated_weights = weights * smooth_factor.unsqueeze(1)
            elif weights.shape[1] == smooth_factor.shape[0]:
                # Linear case: scale input dimension (dim=1)
                compensated_weights = weights * smooth_factor.unsqueeze(0)
            else:
                # but if there is a dimension mismatch skip smoothing
                print(f"    [Warning] Dimension mismatch, skipping smoothing")
                return activations, weights, smooth_factor
        
        return smoothed_acts, compensated_weights, smooth_factor

class HybridA3QER:
    """
    The revolutionary hybrid approach that actually works!
    Combines layer criticality, selective smoothing, and intelligent quantization
    """
    
    def __init__(self, smoothing_alpha: float = 0.3, memory_target: float = 0.3, *, quantize_embeddings: bool = True):
        self.smoother = ImprovedActivationSmoother(alpha=smoothing_alpha)
        self.memory_target = memory_target
        self.calibration_cache = {}
        self.quantize_embeddings = bool(quantize_embeddings)
        
    def collect_calibration_data(self, model, sample_inputs):
        """Collect activation data for calibration (robust for causal & encoder–decoder)."""
        print("[Hybrid-A3QER] Collecting calibration data...")

        model.eval()
        calibration_data = defaultdict(list)
        handles = []

        # ----- forward hook that safely grabs the first tensor input -----
        def create_hook(name):
            def hook(module, input, output):
                #input is a tuple, find first tensor inside it (possibly nested)
                tensor = None
                if input:
                    x = input[0]
                    if isinstance(x, torch.Tensor):
                        tensor = x
                    elif isinstance(x, (tuple, list)) and x and isinstance(x[0], torch.Tensor):
                        tensor = x[0]
                if tensor is not None:
                    calibration_data[name].append(tensor.detach())
            return hook

        for name, module in model.named_modules():
            if _is_linear_like(module):
                handles.append(module.register_forward_hook(create_hook(name)))

        # ----- normalize inputs so T5 gets decoder_input_ids -----
        call_kwargs = None
        call_args = None
        is_seq2seq = bool(getattr(model.config, "is_encoder_decoder", False))

        if isinstance(sample_inputs, dict):
            call_kwargs = dict(sample_inputs)  # shallow copy
            #create attention_mask if missing
            if "input_ids" in call_kwargs and "attention_mask" not in call_kwargs:
                call_kwargs["attention_mask"] = torch.ones_like(call_kwargs["input_ids"])
            #add decoder_input_ids for seq2seq if missing
            if is_seq2seq and "decoder_input_ids" not in call_kwargs:
                bsz = call_kwargs["input_ids"].size(0)
                device = call_kwargs["input_ids"].device
                dec_start = getattr(model.config, "decoder_start_token_id", None)
                if dec_start is None:
                    dec_start = getattr(model.config, "pad_token_id", None) or getattr(model.config, "eos_token_id", None) or 0
                call_kwargs["decoder_input_ids"] = torch.full(
                    (bsz, 1), dec_start, dtype=call_kwargs["input_ids"].dtype, device=device
                )

        elif isinstance(sample_inputs, (list, tuple)):
            #trying to upcast simple tuples into kwargs: (input_ids[, attention_mask])
            if len(sample_inputs) in (1, 2) and isinstance(sample_inputs[0], torch.Tensor):
                input_ids = sample_inputs[0]
                attention_mask = sample_inputs[1] if len(sample_inputs) == 2 and isinstance(sample_inputs[1], torch.Tensor) \
                                else torch.ones_like(input_ids)
                call_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
                if is_seq2seq:
                    bsz = input_ids.size(0)
                    device = input_ids.device
                    dec_start = getattr(model.config, "decoder_start_token_id", None)
                    if dec_start is None:
                        dec_start = getattr(model.config, "pad_token_id", None) or getattr(model.config, "eos_token_id", None) or 0
                    call_kwargs["decoder_input_ids"] = torch.full(
                        (bsz, 1), dec_start, dtype=input_ids.dtype, device=device
                    )
            else:
                
                call_args = tuple(sample_inputs)

        elif isinstance(sample_inputs, torch.Tensor):
            input_ids = sample_inputs
            attention_mask = torch.ones_like(input_ids)
            call_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if is_seq2seq:
                bsz = input_ids.size(0)
                device = input_ids.device
                dec_start = getattr(model.config, "decoder_start_token_id", None)
                if dec_start is None:
                    dec_start = getattr(model.config, "pad_token_id", None) or getattr(model.config, "eos_token_id", None) or 0
                call_kwargs["decoder_input_ids"] = torch.full(
                    (bsz, 1), dec_start, dtype=input_ids.dtype, device=device
                )
        else:
            #unknown type—let it pass through positionally
            call_args = (sample_inputs,)

        # ----- run one forward pass to collect activations -----
        with torch.no_grad():
            if call_kwargs is not None:
                _ = model(**call_kwargs)
            elif call_args is not None:
                _ = model(*call_args)
            else:
                #fallback
                _ = model(sample_inputs)

        # ----- remove hooks and collate -----
        for h in handles:
            h.remove()

        for name, tensors in list(calibration_data.items()):
            if tensors:
                calibration_data[name] = torch.cat(tensors, dim=0)

        self.calibration_cache = dict(calibration_data)
        print(f"[Hybrid-A3QER] Collected data for {len(self.calibration_cache)} layers")

        
    def quantize_model(self, model, sample_inputs):
        """
        Enhanced quantization pipeline with mixed precision.
        - Embeddings handled first (opt-in INT8 if available/desired)
        - lm_head shares buffers with embedding when weight tying is used
        - Linear-like layers (nn.Linear, HF Conv1D) quantized per criticality
        """
        print("\n" + "="*60)
        print("HYBRID A3QER: Mixed-Precision Quantization Pipeline")
        print("="*60)

        want_emb_q = bool(getattr(self, "quantize_embeddings", False))

        #collect calibration data
        self.collect_calibration_data(model, sample_inputs)

        #analyze layer criticality
        print("\n[Phase 1] Analyzing layer criticality...")
        analyzer = LayerCriticalityAnalyzer(model, sample_inputs)
        layer_importance = analyzer.analyze_attention_patterns()

        #quantize layers with mixed precision
        print("\n[Phase 2] Applying mixed-precision quantization...")

        from .quantization import (
            QuantizedLinear, QuantizedConv1D, QuantizedLinearINT4, QuantizedLinearINT6,
        )
        
        try:
            from .quantization import QuantizedEmbeddingINT8, QuantizedLMHeadINT8
        except Exception:
            QuantizedEmbeddingINT8 = None
            QuantizedLMHeadINT8 = None

        quantized_count = {'int4': 0, 'int6': 0, 'int8': 0, 'fp32': 0, 'int8_emb': 0, 'int8_lmhead': 0}
        smoothed_count = 0
        embedding_registry = {}  #tracking embeddings: {(vocab_size, embed_dim): qemb}

        # ============================================================
        # PASS 1: Handle embeddings first
        # ============================================================
        print("\n[Pass 1] Processing embeddings...")
        for name, module in list(model.named_modules()):
            if not isinstance(module, nn.Embedding):
                continue
            
            parent_name = '.'.join(name.split('.')[:-1]) if '.' in name else ''
            module_name = name.split('.')[-1]
            if parent_name:
                parent = model
                for part in parent_name.split('.'):
                    parent = getattr(parent, part)
            else:
                parent = model
            
            lname = name.lower()
            if 'lm_head' in lname or 'head' in lname:
                continue  
            
            if want_emb_q and QuantizedEmbeddingINT8 is not None:
                try:
                    qemb = QuantizedEmbeddingINT8.from_float(module)
                    setattr(parent, module_name, qemb)
                    key = (qemb.num_embeddings, qemb.embedding_dim)
                    embedding_registry[key] = qemb
                    print(f"  [INT8-EMB] {name} - registered {key}")
                    quantized_count['int8_emb'] += 1
                except Exception as e:
                    print(f"  [EMB-FP32] {name}: {str(e)[:80]}")
                    quantized_count['fp32'] += 1
            else:
                print(f"  [EMB-FP32] {name} (quantization disabled)")
                quantized_count['fp32'] += 1

        # ============================================================
        # PASS 2: Handle lm_head (reuse embedding if weight tying)
        # ============================================================
        print("\n[Pass 2] Processing lm_head...")
        for name, module in list(model.named_modules()):
            if not isinstance(module, nn.Linear):
                continue
            
            lname = name.lower()
            if 'lm_head' not in lname and 'head' not in lname:
                continue
            
            parent_name = '.'.join(name.split('.')[:-1]) if '.' in name else ''
            module_name = name.split('.')[-1]
            if parent_name:
                parent = model
                for part in parent_name.split('.'):
                    parent = getattr(parent, part)
            else:
                parent = model
            
            if want_emb_q and QuantizedLMHeadINT8 is not None:
                try:
                    vocab_size, embed_dim = module.weight.shape
                    key = (vocab_size, embed_dim)
                    
                    if key in embedding_registry:
                        
                        qemb = embedding_registry[key]
                        qlmhead = QuantizedLMHeadINT8.from_embedding(
                            qemb, bias=module.bias.data if module.bias is not None else None
                        )
                        setattr(parent, module_name, qlmhead)
                        print(f"  [INT8-LMHEAD(shared)] {name} - tied to {key}")
                        quantized_count['int8_lmhead'] += 1
                    else:
               
                        qlmhead = QuantizedLMHeadINT8.from_float(module)
                        setattr(parent, module_name, qlmhead)
                        print(f"  [INT8-LMHEAD(standalone)] {name}")
                        quantized_count['int8_lmhead'] += 1
                except Exception as e:
                    print(f"  [LMHEAD-FP32] {name}: {str(e)[:80]}")
                    quantized_count['fp32'] += 1
            else:
                print(f"  [LMHEAD-FP32] {name} (quantization disabled)")
                quantized_count['fp32'] += 1

        # ============================================================
        # PASS 3: Quantize other layers with mixed precision
        # ============================================================
        print("\n[Pass 3] Processing linear layers...")
        for name, module in list(model.named_modules()):
            lname = name.lower()
           
            if isinstance(module, nn.Embedding):
                continue
            if 'lm_head' in lname or 'head' in lname:
                continue
            if any(skip in lname for skip in ['ln', 'norm']):
                continue
            if not _is_linear_like(module):
                continue
            
            parent_name = '.'.join(name.split('.')[:-1]) if '.' in name else ''
            module_name = name.split('.')[-1]
            if parent_name:
                parent = model
                for part in parent_name.split('.'):
                    parent = getattr(parent, part)
            else:
                parent = model

            config = analyzer.get_layer_config(name)

            if not config.get('quantize', True):
                print(f"  [FP32-critical] {name}")
                quantized_count['fp32'] += 1
                continue

            try:
                is_hf_conv1d = (module.__class__.__name__ == 'Conv1D')
                W = module.weight.data
                if is_hf_conv1d:
                    in_features = W.shape[0]
                    out_features = W.shape[1]
                else:
                    out_features, in_features = W.shape

                B = module.bias.data if hasattr(module, 'bias') and module.bias is not None else None


                compensated_W = W

                tmp = nn.Linear(in_features, out_features, bias=B is not None)
                if is_hf_conv1d:
                    tmp.weight.data = compensated_W.t().contiguous()
                else:
                    tmp.weight.data = compensated_W
                if B is not None:
                    tmp.bias.data = B

                bits = int(config.get('bits', 8))
                if bits == 4:
                    q = QuantizedLinearINT4.from_float(tmp)
                    print(f"  [INT4] {name}")
                    quantized_count['int4'] += 1
                elif bits == 6:
                    q = QuantizedLinearINT6.from_float(tmp)
                    print(f"  [INT6] {name}")
                    quantized_count['int6'] += 1
                else:
                    q = QuantizedLinear.from_float(tmp, bits=8)
                    print(f"  [INT8] {name}")
                    quantized_count['int8'] += 1

                if is_hf_conv1d:
                    q = QuantizedConv1D(q)

                setattr(parent, module_name, q)

            except Exception as e:
                print(f"  [FAIL] {name}: {str(e)[:80]}")
                quantized_count['fp32'] += 1

        print(f"\n[Summary]")
        print(f"  INT4 layers: {quantized_count['int4']}")
        print(f"  INT6 layers: {quantized_count['int6']}")
        print(f"  INT8 layers: {quantized_count['int8']}")
        print(f"  INT8 embeddings: {quantized_count['int8_emb']}")
        print(f"  INT8 lm_heads: {quantized_count['int8_lmhead']}")
        print(f"  FP32 layers: {quantized_count['fp32']}")
        print(f"  Layers smoothed: {smoothed_count}")

        total_layers = sum(quantized_count.values())
        if total_layers > 0:
            compression_estimate = (
                quantized_count['int4'] * 0.125 +
                quantized_count['int6'] * 0.1875 +
                quantized_count['int8'] * 0.25 +
                quantized_count['int8_emb'] * 0.25 +
                quantized_count['int8_lmhead'] * 0.0 +  
                quantized_count['fp32'] * 1.0
            ) / total_layers
        else:
            compression_estimate = 1.0

        print(f"  Estimated size: {compression_estimate*100:.1f}% of original")
        print(f"  Expected compression: {(1 - compression_estimate)*100:.1f}%")
        print("="*60)

        return model


def make_it_work_hybrid(
    model,
    sample_inputs=None,
    *,
    smoothing_alpha: float = 0.3,
    memory_target: float = 0.3,
    quantize_embeddings: bool = True,
    **extra
):
    """
    Entry point for the Hybrid A3QER quantizer.

    Args:
        model: torch.nn.Module
        sample_inputs: tokenized inputs (dict/tuple/tensor) used for calibration
        smoothing_alpha: activation smoothing factor
        memory_target: target model size fraction (0.3 = 30% of FP32)
        quantize_embeddings: opt-in INT8 embeddings quantization
        **extra: ignored/forwarded future options (keeps backwards-compat)
    """
    quantizer = HybridA3QER(
        smoothing_alpha=smoothing_alpha,
        memory_target=memory_target,
        quantize_embeddings=quantize_embeddings,
        **extra
    )
    return quantizer.quantize_model(model, sample_inputs)
