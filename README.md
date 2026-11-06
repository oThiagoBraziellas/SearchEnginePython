# SearchEngine em Python

Mecanismo de busca de texto completo em Python puro: índice invertido posicional, tokenizador com normalização de acentos, stop words e stemming leve, ranqueamento **BM25**, consultas com frase exata, termos obrigatórios e excluídos, persistência em disco (JSON comprimido, escrita atômica) e uma CLI para indexar arquivos e pesquisar. Vem com uma pequena coleção de documentos de exemplo.

## Por que este projeto

Busca é um problema central em produtos de grande escala (Google, Elasticsearch, Lucene, busca interna de qualquer aplicação) e aparece em entrevistas como "projete um autocomplete", "projete a busca de um site" ou "implemente um índice invertido". Este projeto percorre o caminho completo de um mecanismo real em tamanho pequeno: analisar o texto, montar o índice, pontuar com uma função de ranqueamento moderna, responder a consultas com frases (que exigem posições) e salvar tudo em disco.

## Recursos

- **Tokenizador:** minúsculas, remoção de acentos (`Configuração` encontra `configuracao`), stop words em inglês e português e stemming leve para plurais, `-ing`, `-ed` e `e` final (`caches`, `cached` e `caching` se encontram).
- **Índice invertido posicional:** `termo -> documento -> posições`, com adição, substituição e remoção de documentos.
- **BM25** (Okapi) com `k1` e `b` configuráveis e idf sempre não negativo.
- **Linguagem de consulta:** `cache eviction` (OU ranqueado), `+cache +eviction` (obrigatórios), `-redis` (excluído), `"least recently used"` (frase exata) e `-"frase"`.
- **Frases corretas com stop words:** a posição das palavras removidas é preservada, de modo que `"write ahead"` não casa com `write the ahead`.
- **Trechos (snippets)** em torno do primeiro termo encontrado.
- **Persistência:** JSON comprimido com gzip, posições codificadas por diferenças (delta), gravação em arquivo temporário seguida de `os.replace`; o índice carregado pode ser atualizado e salvo de novo.
- **CLI:** `add`, `remove`, `search`, `stats`.

## Como executar

Requer Python 3.8 ou superior. Dentro da pasta do projeto, usando a coleção de exemplo em `samples/docs`:

```bash
python3 -m search_engine --index exemplo.idx add samples/docs
python3 -m search_engine --index exemplo.idx search 'cache eviction'
python3 -m search_engine --index exemplo.idx search '"consistent hashing"'
python3 -m search_engine --index exemplo.idx search '+log -compaction crash' -n 5
python3 -m search_engine --index exemplo.idx stats
python3 -m search_engine --index exemplo.idx remove samples/docs/caching.txt
```

Como biblioteca:

```python
from search_engine import Index

index = Index()
index.add("doc1", "Consistent hashing places servers on a ring.")
index.add("doc2", "A cache stores expensive results.")
for hit in index.search('"consistent hashing"'):
    print(hit.name, round(hit.score, 3), hit.snippet)
index.save("meu.idx")
index = Index.load("meu.idx")
```

## Como testar

```bash
python3 -m unittest -v
```

Os testes verificam, entre outros pontos: o tokenizador (acentos, offsets, stop words, grupos de stemming); o analisador de consultas; a **fórmula do BM25 contra um valor calculado à mão**, a saturação da frequência de termo, o efeito do tamanho do documento e da raridade do termo; consultas OU/obrigatório/excluído; frases (adjacência, ordem, palavras repetidas, lacunas de stop words); substituição e remoção de documentos; ida e volta pelo disco com resultados idênticos; atualização de um índice carregado; formato desconhecido; e a CLI com a coleção de exemplo.

## Decisões de projeto

- **Índice posicional.** Guardar posições custa mais memória do que guardar só frequências, mas é o que permite frases exatas. A frequência do termo é o tamanho da lista de posições.
- **BM25 em vez de TF-IDF.** A frequência de termo satura (`k1`) e o comprimento do documento é normalizado (`b`), evitando que textos longos ou repetitivos dominem. Idf: `ln(1 + (N - n + 0.5) / (n + 0.5))`.
- **Frase = termos em deslocamentos fixos.** Cada termo da frase carrega seu deslocamento relativo ao primeiro (incluindo lacunas de stop words); verifica-se, por busca binária, se existe um início `p` tal que todos os termos estão em `p + deslocamento`. O custo é proporcional ao número de ocorrências do primeiro termo nos documentos candidatos.
- **Mesmo analisador para documentos e consultas.** É o que garante que `servers` encontre `server`. O stemmer é simples de propósito (e imperfeito, como todo stemmer por regras); trocá-lo é passar outra função ao `Tokenizer`.
- **Candidatos por álgebra de conjuntos.** Obrigatórios e frases fazem interseção, opcionais fazem união, excluídos fazem diferença. Só os candidatos são pontuados.
- **Desempate determinístico.** Pontuações iguais são ordenadas pelo nome do documento.
- **Persistência atômica e compacta.** Posições por diferença comprimem bem com gzip; o arquivo temporário evita que uma queda no meio da gravação corrompa o índice existente.
- **Complexidade:** indexar um documento é O(tokens); consulta é O(soma dos tamanhos das listas de postagens dos termos) mais a ordenação dos candidatos; memória O(total de ocorrências).
- **Trade-off:** o texto original fica guardado no índice para gerar trechos e permitir a remoção; isso aumenta o tamanho do arquivo.

## Estrutura de pastas

```
SearchEngine/
├── search_engine/
│   ├── __init__.py
│   ├── __main__.py
│   ├── tokenizer.py    # análise de texto e stemmer
│   ├── query.py        # linguagem de consulta
│   ├── index.py        # índice, BM25, frases, persistência
│   └── cli.py
├── samples/
│   └── docs/           # seis documentos de exemplo
├── tests/
│   ├── test_tokenizer.py
│   ├── test_index.py
│   └── test_cli.py
├── LICENSE
└── README.md
```

## Limitações conhecidas

- Todo o índice fica na memória e é regravado por inteiro a cada `save`; não há segmentos nem atualização incremental em disco.
- Sem busca por prefixo, curingas, tolerância a erros de digitação (fuzzy) nem sinônimos.
- O stemmer é apenas para inglês, e as stop words cobrem só inglês e português.
- Sem pesos por campo (título vs. corpo): cada documento é um único texto.
- Um único processo por arquivo de índice, sem controle de concorrência.
- A busca percorre todos os candidatos, sem otimizações como WAND ou listas com saltos.

## Licença

MIT. Veja o arquivo [LICENSE](LICENSE).
