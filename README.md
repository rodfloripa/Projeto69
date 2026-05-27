# 1. Projeto: Inferência Otimizada de LLM com Offloading Automático CPU ↔ GPU

<div align="justify">

Este projeto implementa um pipeline moderno de inferência otimizada para Large Language Models (LLMs) utilizando TinyLlama 1.1B com foco em execução eficiente em hardware limitado.

O principal objetivo do projeto foi executar uma LLM moderna utilizando pouca VRAM através de:

* quantização 4-bit
* FP16
* KV Cache
* Flash Attention
* offloading automático CPU ↔ GPU
* gerenciamento dinâmico de memória

O projeto foi desenvolvido inicialmente como uma tentativa de implementar layer-wise inference manualmente, movendo explicitamente cada layer entre CPU e GPU.

Entretanto, durante o desenvolvimento surgiram diversos desafios internos relacionados à arquitetura moderna do Llama, principalmente:

* rotary embeddings
* apply_rotary_pos_emb
* causal masks
* KV cache
* gerenciamento interno de atenção
* sincronização CPU/GPU

A solução final foi utilizar o runtime moderno do Hugging Face Accelerate para delegar automaticamente:

* gerenciamento de memória
* streaming CPU ↔ GPU
* dispatch de layers
* KV cache
* rotary embeddings

O resultado final foi um sistema extremamente eficiente, capaz de executar TinyLlama utilizando menos de 1 GB de VRAM em uma NVIDIA T4.

</div>

---

# 2. Tecnologias Utilizadas

<div align="justify">

O projeto foi construído utilizando:

* PyTorch
* Hugging Face Transformers
* Accelerate
* BitsAndBytes
* CUDA
* FP16
* Quantização NF4
* KV Cache
* Flash Attention

Modelo utilizado:

* TinyLlama 1.1B Chat

</div>

---

# 3. Objetivo Principal do Projeto

<div align="justify">

O principal conceito arquitetural deste projeto foi utilizar memória de forma inteligente.

Ao invés de manter todos os pesos permanentemente carregados na GPU, o runtime utiliza:

* VRAM como cache rápido
* RAM como armazenamento auxiliar
* streaming automático CPU ↔ GPU
* carregamento dinâmico sob demanda

Na prática:

* parte do modelo permanece na RAM
* apenas partes necessárias vão para GPU
* a VRAM é constantemente reutilizada
* o runtime controla automaticamente o offloading

Isso permite executar modelos relativamente grandes em GPUs pequenas.

</div>

---

# 4. Instalação das Dependências

<div align="justify">

O ambiente utiliza bibliotecas modernas para inferência otimizada.

</div>

```bash
!pip install -U transformers -q
!pip install -U bitsandbytes>=0.46.1 -q
!pip install -U accelerate -q
```

<div align="justify">

Cada biblioteca possui uma função específica:

* `transformers` → runtime principal da LLM
* `bitsandbytes` → quantização 4-bit
* `accelerate` → offloading automático CPU/GPU

</div>

---

# 5. Configuração Principal

<div align="justify">

O primeiro bloco define as configurações centrais da inferência.

</div>

```python
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

MAX_NEW_TOKENS = 120

TEMPERATURE = 0.8
TOP_P = 0.95
TOP_K = 50

USE_4BIT = True

PROMPT = """
Explain why transformers consume a lot of memory.
"""
```

<div align="justify">

Esses parâmetros controlam:

* modelo utilizado
* quantidade máxima de tokens
* diversidade textual
* ativação da quantização
* prompt de entrada

</div>

---

# 6. Otimizações CUDA

<div align="justify">

O projeto ativa otimizações modernas da NVIDIA para acelerar inferência e reduzir overhead computacional.

</div>

```python
if DEVICE == "cuda":

    print(torch.cuda.get_device_name(0))

    torch.backends.cuda.matmul.allow_tf32 = True

    torch.set_float32_matmul_precision("high")

    torch.backends.cuda.enable_flash_sdp(True)

    torch.backends.cuda.enable_mem_efficient_sdp(True)

    torch.cuda.empty_cache()
```

<div align="justify">

Essas otimizações ativam:

* Tensor Cores
* TF32
* Flash Attention kernels
* memória eficiente para self-attention

O objetivo é reduzir:

* uso de VRAM
* latência inferencial
* movimentação de memória

</div>

---

# 7. Quantização 4-bit

<div align="justify">

A quantização foi um dos pilares centrais do projeto.

</div>

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=False
)
```

<div align="justify">

A quantização converte os pesos do modelo para representações compactas de 4 bits.

Comparação aproximada:

| Precisão | Bits por parâmetro |
| -------- | ------------------ |
| FP32     | 32                 |
| FP16     | 16                 |
| NF4      | 4                  |

Isso reduz drasticamente:

* uso de VRAM
* uso de RAM
* bandwidth de memória

Sem quantização, o consumo de memória seria muito maior.

</div>

---

# 8. Tokenizer

<div align="justify">

O tokenizer converte texto em tokens inteiros utilizados pela LLM.

</div>

```python
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

if tokenizer.pad_token is None:

    tokenizer.pad_token = tokenizer.eos_token
```

<div align="justify">

O `pad_token` foi ajustado para evitar problemas durante geração autoregressiva.

</div>

---

# 9. Carregamento do Modelo

<div align="justify">

O ponto mais importante do projeto está neste bloco:

</div>

```python
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True
)
```

<div align="justify">

O parâmetro mais importante é:

</div>

```python
device_map="auto"
```

<div align="justify">

Ele ativa gerenciamento automático de memória através do Accelerate.

Internamente o runtime:

* distribui layers entre CPU e GPU
* faz offloading automático
* reutiliza VRAM dinamicamente
* streama pesos sob demanda
* evita manter o modelo inteiro na GPU

Na prática:

* GPU funciona como cache rápido
* RAM funciona como memória auxiliar

Esse mecanismo é conhecido como:

</div>

```text
memory offloading
```

<div align="justify">

e também:

</div>

```text
memory streaming
```

<div align="justify">

Essa foi a principal ideia arquitetural do projeto.

</div>

---

# 10. Inferência FP16

<div align="justify">

Outro ponto fundamental foi utilizar FP16.

</div>

```python
torch_dtype=torch.float16
```

<div align="justify">

Isso reduz pela metade o tamanho das ativações internas.

Durante inferência, não apenas os pesos consomem memória.

As ativações da self-attention também possuem alto custo.

FP16 reduz:

* consumo de memória
* bandwidth GPU
* latência inferencial

Além disso, ativa Tensor Cores da GPU NVIDIA.

</div>

---

# 11. Tokenização do Prompt

<div align="justify">

O prompt é convertido para tensores PyTorch.

</div>

```python
inputs = tokenizer(
    PROMPT,
    return_tensors="pt"
)

input_ids = inputs["input_ids"].to(DEVICE)

attention_mask = inputs["attention_mask"].to(DEVICE)
```

<div align="justify">

Esses tensores representam:

* tokens da sequência
* máscara causal da atenção

</div>

---

# 12. KV Cache

<div align="justify">

Outro componente extremamente importante foi:

</div>

```python
use_cache=True
```

<div align="justify">

Transformers normalmente possuem complexidade:

</div>

O(n^2)

<div align="justify">

porque a atenção precisa recalcular toda a sequência anterior.

O KV Cache evita recomputar:

* Keys
* Values

já processados anteriormente.

Isso reduz drasticamente:

* computação
* uso de memória temporária
* latência

Sem KV Cache a inferência seria muito mais lenta.

</div>

---

# 13. Flash Attention

<div align="justify">

O runtime moderno também utiliza Flash Attention automaticamente quando disponível.

A ideia principal é evitar materializar completamente a matriz de atenção:

</div>

QK^T

<div align="justify">

Flash Attention computa a atenção em blocos menores diretamente na SRAM da GPU.

Isso reduz:

* uso de VRAM
* leitura/escrita em HBM
* tensores intermediários

Os ganhos incluem:

* maior throughput
* menor consumo de memória
* melhor escalabilidade

</div>

---

# 14. Geração Autoregressiva

<div align="justify">

A geração foi realizada utilizando o runtime oficial do Transformers.

</div>

```python
outputs = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_new_tokens=MAX_NEW_TOKENS,
    do_sample=True,
    temperature=TEMPERATURE,
    top_p=TOP_P,
    top_k=TOP_K,
    repetition_penalty=1.1,
    use_cache=True,
    pad_token_id=tokenizer.eos_token_id
)
```

<div align="justify">

O runtime gerencia automaticamente:

* rotary embeddings
* KV cache
* causal attention
* sampling
* decoding
* offloading CPU/GPU

Isso eliminou diversos erros encontrados durante as tentativas iniciais de implementar layer-wise inference manualmente.

</div>

---

# 15. Decodificação do Texto

<div align="justify">

Após a geração, os tokens são convertidos novamente para texto.

</div>

```python
generated_text = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)
```

<div align="justify">

Isso transforma os IDs inteiros produzidos pela LLM em linguagem natural.

</div>

---

# 16. Benchmark de Memória

<div align="justify">

O projeto também mede consumo de VRAM.

</div>

```python
peak = torch.cuda.max_memory_allocated() / 1024**3

print(f"Peak GPU Memory: {peak:.2f} GB")
```

<div align="justify">

Isso permitiu validar a eficiência do offloading automático.

</div>

---

# 17. Problemas Encontrados Durante o Desenvolvimento

<div align="justify">

Durante o desenvolvimento surgiram vários problemas internos relacionados à arquitetura moderna do Llama.

Os principais erros envolveram:

* `apply_rotary_pos_emb`
* `position_embeddings`
* `NoneType unpacking`
* mismatch CPU/GPU
* dimension mismatch
* causal masks
* KV cache
* sincronização de dispositivos

A tentativa inicial era implementar manualmente:

* layer streaming
* rotary embeddings
* KV cache
* dispatch de layers

Entretanto, os runtimes modernos do Llama possuem muitos detalhes internos complexos.

A solução correta foi delegar:

* rotary embeddings
* causal masks
* KV cache
* offloading
* gerenciamento de memória

para o runtime oficial do Hugging Face.

</div>

---

# 18. Resultado Final

<div align="justify">

O resultado final foi extremamente eficiente.

</div>

```text
Generation Time: 5.61s
Peak GPU Memory: 0.87 GB
```

<div align="justify">

Isso demonstrou que:

* quantização funcionou corretamente
* offloading automático funcionou
* KV cache estava ativo
* Flash Attention estava ativo
* gerenciamento dinâmico de memória funcionou

O ponto mais impressionante foi o baixo consumo de VRAM:

</div>

```text
0.87 GB
```

<div align="justify">

para uma LLM de 1.1B parâmetros.

</div>

---

# 19. Exemplo de Texto Gerado

<div align="justify">

O modelo conseguiu gerar texto semanticamente coerente.

</div>

```text
Explain why transformers consume a lot of memory.

Memory usage in a transformer is dependent on the model size and the number of layers...
```

<div align="justify">

Isso confirmou que:

* embeddings estavam corretos
* self-attention estava funcionando
* geração autoregressiva estava estável
* o runtime estava gerenciando corretamente o modelo

</div>

---

# 20. Conclusão

<div align="justify">

Este projeto demonstrou como executar Large Language Models modernas em hardware relativamente limitado utilizando técnicas modernas de inferência otimizada.

O principal aprendizado não foi apenas sobre Transformers, mas sobre engenharia de runtime para LLMs.

Hoje, grande parte da eficiência de inferência depende de:

* quantização
* memory streaming
* CPU/GPU offloading
* KV cache
* Flash Attention
* gerenciamento dinâmico de memória

O projeto também mostrou que frameworks modernos como Hugging Face Accelerate já implementam internamente mecanismos extremamente sofisticados de:

* dispatch de layers
* balancing de memória
* offloading automático
* streaming de pesos
* gerenciamento de dispositivos

A implementação final ficou muito próxima da filosofia utilizada em runtimes profissionais modernos de serving de LLMs.

</div>
