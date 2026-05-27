# 1. Projeto: Inferência Otimizada de LLM com Streaming de Memória CPU ↔ GPU

<div align="justify">

Este projeto implementa um pipeline completo de inferência otimizada para Large Language Models (LLMs) utilizando TinyLlama 1.1B com foco em execução eficiente em hardware limitado.

O principal objetivo não foi apenas executar uma LLM localmente, mas construir um sistema de inferência capaz de:

* utilizar pouca VRAM
* aproveitar RAM como memória auxiliar
* fazer streaming dinâmico de pesos
* reduzir uso de memória GPU
* manter boa velocidade de geração

O projeto utiliza quantização 4-bit, inferência FP16, KV Cache e offloading automático CPU ↔ GPU através do runtime moderno do Hugging Face Accelerate.

O resultado final foi um sistema capaz de executar uma LLM moderna consumindo menos de 1 GB de VRAM em uma NVIDIA T4.

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
* Quantização NF4
* FP16
* KV Cache
* Flash Attention

Modelo utilizado:

* TinyLlama 1.1B Chat

</div>

---

# 3. Objetivo Arquitetural do Projeto

<div align="justify">

O principal diferencial deste projeto foi implementar inferência orientada a memória.

Ao invés de manter todos os pesos permanentemente carregados na GPU, o sistema utiliza:

* VRAM como cache rápido
* RAM como armazenamento auxiliar
* streaming dinâmico CPU ↔ GPU
* carregamento sob demanda

Essa abordagem permite executar modelos relativamente grandes em GPUs pequenas.

A filosofia principal do projeto foi:

</div>

```text
carregar apenas o necessário na GPU
executar
liberar memória
continuar o streaming
```

<div align="justify">

Esse conceito é utilizado em runtimes modernos como:

* vLLM
* DeepSpeed ZeRO-Inference
* llama.cpp
* Ollama
* TensorRT-LLM
* Hugging Face Accelerate

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
!pip install -U huggingface_hub -q
```

<div align="justify">

Cada biblioteca possui uma função específica:

* `transformers` → runtime principal do modelo
* `bitsandbytes` → quantização 4-bit
* `accelerate` → streaming CPU/GPU
* `huggingface_hub` → autenticação e download

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

SAVE_PATH = "./tinyllama_local"
```

<div align="justify">

Esses parâmetros controlam:

* modelo carregado
* diversidade textual
* tamanho máximo da geração
* ativação de quantização
* armazenamento local do modelo

</div>

---

# 6. Login no Hugging Face

<div align="justify">

O projeto suporta autenticação via Hugging Face Token.

</div>

```python
from huggingface_hub import login

HF_TOKEN = "SEU_TOKEN"

login(HF_TOKEN)
```

<div align="justify">

Isso permite:

* download automático
* acesso a modelos privados
* autenticação persistente

</div>

---

# 7. Quantização 4-bit

<div align="justify">

A quantização foi um dos pilares principais do projeto.

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

Sem quantização, mesmo TinyLlama consumiria muito mais memória.

</div>

---

# 8. Streaming de Memória CPU ↔ GPU

<div align="justify">

O conceito mais importante do projeto está neste bloco:

</div>

```python
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map="auto"
)
```

<div align="justify">

O parâmetro mais importante aqui é:

</div>

```python
device_map="auto"
```

<div align="justify">

Ele ativa gerenciamento automático de memória através do Accelerate.

Internamente o runtime faz algo parecido com:

</div>

```text
Layer 1 → GPU
executa
remove

Layer 2 → GPU
executa
remove

Layer 3 → GPU
executa
remove
```

<div align="justify">

Isso significa que:

* parte do modelo permanece na RAM
* apenas partes necessárias vão para GPU
* VRAM é constantemente liberada
* pesos são streamados dinamicamente

Na prática:

* GPU funciona como cache rápido
* RAM funciona como armazenamento auxiliar

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

# 9. Inferência FP16

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

# 10. Salvamento Local do Modelo

<div align="justify">

O projeto também salva os pesos localmente após o primeiro download.

</div>

```python
if not os.path.exists(SAVE_PATH):

    model.save_pretrained(SAVE_PATH)

    tokenizer.save_pretrained(SAVE_PATH)
```

<div align="justify">

Isso evita:

* redownload
* tráfego desnecessário
* dependência constante da internet

Depois disso o carregamento ocorre diretamente do disco.

</div>

---

# 11. Carregamento Inteligente

<div align="justify">

O carregamento verifica automaticamente se o modelo já existe localmente.

</div>

```python
if os.path.exists(SAVE_PATH):

    tokenizer = AutoTokenizer.from_pretrained(
        SAVE_PATH
    )

    model = AutoModelForCausalLM.from_pretrained(
        SAVE_PATH,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map="auto"
    )

else:

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map="auto"
    )
```

<div align="justify">

Esse mecanismo melhora significativamente:

* inicialização
* estabilidade
* velocidade de uso

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

A geração foi realizada utilizando `model.generate()`.

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

Isso eliminou diversos erros encontrados durante tentativas manuais de layer-wise inference.

</div>

---

# 15. Problemas Encontrados

<div align="justify">

Durante o desenvolvimento surgiram vários problemas relacionados a:

* rotary embeddings
* position ids
* causal masks
* KV cache
* mismatch CPU/GPU
* apply_rotary_pos_emb

A tentativa inicial era implementar layer streaming manualmente.

Entretanto, os runtimes modernos do Llama possuem muitos detalhes internos complexos.

A solução correta foi delegar:

* rotary embeddings
* KV cache
* causal masks
* offloading

para o runtime oficial do Hugging Face.

</div>

---

# 16. Resultado Final

<div align="justify">

O resultado final foi extremamente eficiente.

</div>

```text
Generation Time: 5.61s
Peak GPU Memory: 0.87 GB
```

<div align="justify">

Isso demonstra que:

* quantização funcionou corretamente
* streaming CPU/GPU funcionou
* offloading automático funcionou
* KV cache estava ativo
* Flash Attention estava ativo

O mais impressionante foi o baixo consumo de VRAM:

</div>

```text
0.87 GB
```

<div align="justify">

para uma LLM de 1.1B parâmetros.

</div>

---

# 17. Exemplo de Texto Gerado

<div align="justify">

O modelo conseguiu gerar texto semanticamente coerente.

</div>

```text
Explain why transformers consume a lot of memory.

Memory usage in a transformer is dependent on the model size and the number of layers...
```

<div align="justify">

Isso confirmou que:

* attention estava correta
* embeddings estavam corretos
* geração autoregressiva estava estável

</div>

---

# 18. Conclusão

<div align="justify">

Este projeto demonstrou na prática como executar Large Language Models modernas em hardware relativamente limitado utilizando técnicas avançadas de gerenciamento de memória.

O principal aprendizado não foi apenas sobre Transformers, mas sobre engenharia de runtime de inferência.

Hoje, grande parte da eficiência de LLMs depende de:

* memory streaming
* CPU/GPU offloading
* quantização
* KV cache
* Flash Attention
* gerenciamento dinâmico de memória

A implementação final mostrou que GPUs pequenas conseguem executar modelos modernos de forma extremamente eficiente quando o runtime correto é utilizado.

O projeto também evidenciou que frameworks modernos como Hugging Face Accelerate já implementam internamente mecanismos extremamente sofisticados de:

* layer dispatch
* memory balancing
* streaming de pesos
* gerenciamento automático de dispositivos

A arquitetura final ficou muito próxima dos runtimes profissionais utilizados atualmente para serving de LLMs em produção.

</div>
