# 1. Projeto: Inferência Otimizada de LLM com TinyLlama e Quantização 4-bit

<div align="justify">

Este projeto demonstra como executar uma Large Language Model (LLM) de forma eficiente utilizando técnicas modernas de otimização de inferência. O modelo utilizado foi o TinyLlama 1.1B, executado em GPU NVIDIA T4 através do Hugging Face Transformers com quantização 4-bit NF4.

O objetivo principal foi construir um pipeline de inferência leve, rápido e com baixo consumo de memória VRAM, mantendo qualidade textual adequada para geração autoregressiva.

Durante o desenvolvimento, vários problemas internos relacionados a rotary embeddings, attention masks e layer-wise inference foram encontrados e corrigidos, resultando em uma implementação estável e eficiente.

</div>

---

# 2. Tecnologias Utilizadas

<div align="justify">

O projeto utiliza principalmente:

* PyTorch
* Hugging Face Transformers
* BitsAndBytes
* CUDA
* Quantização NF4
* Inferência FP16
* KV Cache
* Flash Attention

A arquitetura base utilizada foi:

* TinyLlama 1.1B Chat

</div>

---

# 3. Objetivo do Projeto

<div align="justify">

O foco do projeto foi investigar técnicas de otimização de inferência para LLMs locais, especialmente:

* Redução de VRAM
* Aceleração de geração de texto
* Execução em GPUs pequenas
* Quantização agressiva
* Uso correto do runtime do Hugging Face

Inicialmente tentou-se implementar layer-wise inference manualmente, incluindo manipulação explícita de rotary embeddings. Entretanto, essa abordagem gerou diversos erros internos relacionados às implementações modernas do Llama.

A solução final adotada foi utilizar o runtime oficial do Hugging Face corretamente, permitindo que o próprio framework gerenciasse:

* Rotary Positional Embeddings (RoPE)
* KV Cache
* Attention Masks
* Flash Attention
* Position IDs
* Otimizações CUDA

</div>

---

# 4. Configuração do Modelo

<div align="justify">

O primeiro bloco importante define as configurações principais da inferência.

</div>

```python
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

MAX_NEW_TOKENS = 120

TEMPERATURE = 0.8
TOP_P = 0.95
TOP_K = 50

USE_4BIT = True
```

<div align="justify">

Aqui são definidos:

* O modelo utilizado
* Quantidade máxima de tokens gerados
* Parâmetros de sampling
* Uso de quantização 4-bit

Os parâmetros `temperature`, `top_p` e `top_k` controlam diversidade textual durante a geração.

</div>

---

# 5. Quantização 4-bit com BitsAndBytes

<div align="justify">

O bloco abaixo implementa quantização NF4.

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

Essa etapa foi fundamental para reduzir drasticamente o consumo de VRAM.

A quantização converte os pesos do modelo para representações compactas de 4 bits, mantendo boa qualidade inferencial.

A configuração NF4 é atualmente uma das mais eficientes para inferência de LLMs.

Os ganhos incluem:

* Redução extrema de memória
* Maior throughput
* Menor uso de VRAM
* Possibilidade de executar modelos maiores

</div>

---

# 6. Bloco Principal de Otimização de Memória

<div align="justify">

O principal bloco responsável pela otimização de memória foi:

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

Esse trecho concentra praticamente todas as otimizações importantes do projeto.

Cada parâmetro possui impacto direto no uso de RAM e VRAM.

</div>

---

## 6.1 Quantização dos Pesos

<div align="justify">

O parâmetro:

</div>

```python
quantization_config=bnb_config
```

<div align="justify">

faz com que os pesos originais do modelo deixem de ser armazenados em FP16 ou FP32.

Normalmente:

* FP32 usa 32 bits por parâmetro
* FP16 usa 16 bits por parâmetro
* NF4 usa apenas 4 bits

Isso reduz drasticamente o footprint de memória.

Um modelo de 1.1B parâmetros em FP16 pode consumir vários gigabytes de VRAM, enquanto em 4-bit o consumo cai para menos de 1 GB.

Além disso, o BitsAndBytes implementa kernels otimizados para:

* descompressão dinâmica
* multiplicação matricial quantizada
* streaming eficiente de pesos

Isso permite que o modelo continue relativamente rápido mesmo extremamente comprimido.

</div>

---

## 6.2 Execução FP16

<div align="justify">

O parâmetro:

</div>

```python
torch_dtype=torch.float16
```

<div align="justify">

faz com que ativações intermediárias utilizem half precision.

Esse ponto é extremamente importante porque durante inferência o maior custo nem sempre são os pesos do modelo.

As ativações internas da self-attention podem consumir enormes quantidades de memória.

Utilizando FP16:

* cada tensor ocupa metade da memória
* kernels CUDA ficam mais rápidos
* Tensor Cores da GPU são utilizados
* bandwidth de memória é reduzido

Na prática, FP16 reduz significativamente:

* uso de VRAM
* tráfego de memória
* latência inferencial

</div>

---

## 6.3 Device Map Automático

<div align="justify">

O parâmetro:

</div>

```python
device_map="auto"
```

<div align="justify">

foi um dos pontos mais importantes do projeto.

Inicialmente tentou-se mover layers manualmente entre CPU e GPU, mas isso gerou diversos erros internos envolvendo:

* rotary embeddings
* KV cache
* position ids
* causal masks
* tensores em dispositivos incorretos

Com `device_map="auto"`, o próprio Hugging Face gerencia:

* distribuição de layers
* movimentação CPU/GPU
* carregamento eficiente
* offloading automático
* memória disponível

Isso evita:

* cópias desnecessárias
* fragmentation de VRAM
* sincronizações incorretas
* device mismatch

Além disso, o runtime consegue aplicar otimizações internas invisíveis ao usuário.

</div>

---

## 6.4 KV Cache

<div align="justify">

Outro ponto fundamental da otimização foi:

</div>

```python
use_cache=True
```

<div align="justify">

durante a geração.

Transformers normalmente recalculariam toda a sequência anterior para cada novo token.

Isso gera complexidade quadrática:

</div>

O(n^2)

<div align="justify">

O KV cache evita isso armazenando:

* keys
* values

já computados anteriormente.

Assim, cada novo token reutiliza os estados anteriores sem recomputar toda a atenção causal.

Os ganhos incluem:

* enorme aceleração de inferência
* redução computacional
* menor uso de memória temporária
* menor latência

Sem KV cache, a geração seria muito mais lenta.

</div>

---

## 6.5 Flash Attention

<div align="justify">

Mesmo sem configuração explícita, o runtime moderno do Hugging Face ativa automaticamente kernels otimizados conhecidos como Flash Attention quando disponíveis.

Flash Attention reduz drasticamente:

* movimentação de memória
* leitura/escrita em HBM
* criação de tensores intermediários

A ideia principal é evitar materializar completamente a matriz de atenção.

Normalmente a atenção possui custo:

</div>

QK^T

<div align="justify">

o que gera matrizes enormes para sequências grandes.

Flash Attention computa a atenção em blocos menores diretamente na SRAM da GPU, reduzindo drasticamente o consumo de memória.

Isso melhora:

* throughput
* estabilidade
* escalabilidade
* eficiência energética

</div>

---

# 7. Tokenização

<div align="justify">

A entrada textual é convertida em tokens através do tokenizer.

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

Essa etapa transforma texto em IDs inteiros compatíveis com o Transformer.

O `attention_mask` informa ao modelo quais posições devem ser consideradas durante a atenção causal.

</div>

---

# 8. Geração Autoregressiva

<div align="justify">

A geração textual foi realizada utilizando `model.generate()`.

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

Esse bloco é o núcleo da inferência.

Aqui o Hugging Face gerencia automaticamente:

* KV Cache
* Causal Attention
* Rotary Embeddings
* Sampling
* Decoding autoregressivo

O parâmetro mais importante é:

</div>

```python
use_cache=True
```

<div align="justify">

Ele permite reutilizar chaves e valores anteriores da self-attention, acelerando drasticamente a geração.

Sem KV cache, o modelo precisaria recomputar toda a sequência a cada novo token.

</div>

---

# 9. Problemas Encontrados Durante o Desenvolvimento

<div align="justify">

Diversos problemas internos apareceram ao tentar implementar inferência layer-wise manualmente.

Os principais erros envolveram:

* `apply_rotary_pos_emb`
* incompatibilidade de dimensões
* tensores CPU/GPU
* causal masks incorretas
* ausência de KV cache
* mismatch entre head_dim e rotary_dim

O principal aprendizado foi:

</div>

```text
Não reimplementar internals do Llama manualmente.
```

<div align="justify">

As implementações modernas do Hugging Face possuem muitos detalhes internos altamente específicos e otimizados.

Manipular rotary embeddings manualmente causava:

* geração corrompida
* texto aleatório
* RuntimeErrors
* desalinhamento posicional

A solução correta foi delegar toda a lógica interna ao runtime oficial do Transformers.

</div>

---

# 10. Resultado Final

<div align="justify">

O resultado final foi extremamente eficiente.

</div>

```text
Generation Time: 5.61s
Peak GPU Memory: 0.87 GB
```

<div align="justify">

Isso demonstra que:

* A quantização 4-bit funcionou corretamente
* O KV cache estava ativo
* A inferência FP16 estava funcionando
* O runtime CUDA foi otimizado corretamente

O consumo de apenas 0.87 GB de VRAM é extremamente baixo para uma LLM de 1.1B parâmetros.

</div>

---

# 11. Texto Gerado pelo Modelo

<div align="justify">

O modelo conseguiu produzir texto coerente e semanticamente correto.

Exemplo:

</div>

```text
Explain why transformers consume a lot of memory.

Step 4: Memory Usage

Memory usage in a transformer is dependent on the model size and the number of layers...
```

<div align="justify">

Isso confirmou que:

* Attention estava correta
* Position embeddings estavam corretos
* O modelo manteve coerência textual
* A geração autoregressiva estava funcionando adequadamente

</div>

---

# 12. Conclusão

<div align="justify">

Este projeto mostrou na prática como executar LLMs modernas de forma eficiente utilizando quantização e otimizações de inferência.

O maior aprendizado não foi sobre a arquitetura Transformer em si, mas sobre engenharia de runtime de inferência.

Hoje, grande parte da performance de LLMs depende de:

* gerenciamento de memória
* caching
* movemento CPU/GPU
* kernels CUDA
* quantização
* runtime de serving

A implementação final mostrou que mesmo GPUs relativamente simples, como a NVIDIA T4, conseguem executar modelos modernos de forma extremamente eficiente quando o pipeline correto é utilizado.

O projeto também evidenciou a complexidade interna dos runtimes modernos de LLMs, especialmente em relação a:

* rotary embeddings
* KV cache
* attention internals
* Flash Attention
* gerenciamento automático de dispositivos

A versão final alcançou um pipeline robusto, rápido e estável para inferência local de LLMs.

</div>
