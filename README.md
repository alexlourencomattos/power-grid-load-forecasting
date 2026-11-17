# Power Grid Load Forecasting

[![CI](https://github.com/alexlourencomattos/power-grid-load-forecasting/actions/workflows/ci.yml/badge.svg)](https://github.com/alexlourencomattos/power-grid-load-forecasting/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Production-ready data engineering and machine learning pipeline for seven-day
hourly electricity load forecasting across Brazil's interconnected power grid.

O projeto usa dados abertos do Operador Nacional do Sistema Elétrico (ONS) para
extrair, validar e consolidar a curva de carga horária brasileira. A saída cobre
os subsistemas Norte, Nordeste, Sul e Sudeste/Centro-Oeste, calcula a carga do
Sistema Interligado Nacional (SIN) e produz previsões para 168 horas.

> Estado do projeto: pipeline, treinamento temporal, seleção contra baseline,
> previsão de 168 horas e API estão implementados e validados com dados reais.

## Fonte oficial

- Catálogo: <https://dados.ons.org.br/dataset/curva-carga>
- API CKAN usada na descoberta: `package_show?id=curva-carga`
- Granularidade: horária
- Medida: carga de energia em MWmed
- Atualização informada pelo catálogo: diária
- Licença da fonte: Creative Commons Atribuição

Os arquivos do ONS podem ser corrigidos depois da publicação. Por isso, a
pipeline redescobre os recursos no catálogo e substitui integralmente os anos
solicitados, mantendo cada versão bruta identificada por SHA-256.

## Arquitetura

```text
API CKAN do ONS
      |
      v
bronze: CSV original + SHA-256
      |
      v
silver: Parquet normalizado por instante e subsistema
      |
      v
gold: Parquet horário com N, NE, S, SE e total SIN
      |
      v
atributos sem vazamento -> backtest -> modelo versionado
      |
      v
previsão Parquet de 168 horas -> API HTTP
```

Os dados gerados ficam em:

```text
data/
├── bronze/ons_load/year=AAAA/     # arquivos originais imutáveis
├── silver/ons_load_hourly.parquet # série longa por subsistema
├── gold/ons_load_hourly.parquet   # série larga e total do SIN
└── manifests/                     # auditoria de cada execução
```

O manifesto registra recursos, URLs, datas de atualização, hashes, quantidade
de linhas, caminhos e resultado da execução. Dados gerados ficam fora do Git.

## Requisitos e instalação

- Python 3.11 ou superior
- acesso HTTPS a `dados.ons.org.br` e ao armazenamento indicado pelo catálogo

No Windows (PowerShell):

```powershell
cd "C:\Users\Alex Lourenço\PycharmProjects\machinelearning"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Linux e macOS usam `source .venv/bin/activate` para ativar o ambiente.

## Executar a pipeline

Carga inicial recomendada para desenvolver o modelo:

```powershell
ons-load --start-year 2019 --end-year 2026
```

Ou sem usar o comando instalado:

```powershell
python -m ons_load_pipeline.cli --start-year 2019 --end-year 2026
```

Para a atualização diária, reprocesse o ano atual e o anterior. Isso incorpora
correções tardias do ONS:

```powershell
ons-load --start-year 2025 --end-year 2026
```

Use `--data-dir D:\ons-data` para guardar as camadas em outro local. Um arquivo
de lock impede duas execuções simultâneas no mesmo diretório. Se o processo for
encerrado abruptamente, confirme que não existe uma pipeline ativa antes de
remover `data/.ons_load_pipeline.lock`.

## Treinar e gerar previsões

Faça a carga histórica inicial antes do primeiro treino:

```powershell
ons-load --start-year 2019 --end-year 2026
ons-train --validation-days 56
ons-forecast
```

O treinamento cria um artefato versionado, `artifacts/models/latest.joblib` e
um relatório JSON com métricas. A previsão cria 168 linhas horárias em
`artifacts/forecasts/latest.parquet`, além de preservar a versão identificada
pela origem da previsão.

Para executar atualização, treino e previsão em sequência:

```powershell
ons-production
```

Por padrão, a rotina diária reprocessa o ano atual e o anterior, treina usando
todo o histórico já existente na gold e publica uma nova previsão. Na primeira
execução, faça antes a carga histórica de 2019 em diante.

### Metodologia do modelo

São treinados quatro `HistGradientBoostingRegressor`, um por subsistema. Cada
linha representa uma hora de um horizonte de 1 a 168 horas. Os atributos usam
somente dados disponíveis na origem da previsão:

- hora, dia da semana e posição anual em codificação cíclica;
- cargas da mesma hora há uma, duas e três semanas;
- variação entre semanas e média das três semanas;
- carga observada 1, 24 e 168 horas antes da origem;
- número da hora dentro do horizonte.

O modelo aprende o resíduo sobre a mesma hora da semana anterior. As últimas
oito semanas são reservadas para validação temporal. Para cada subsistema, a
produção escolhe o modelo apenas quando seu MAE é menor que o baseline semanal.
O SIN é calculado pela soma das quatro previsões, garantindo coerência.

### Resultado do backtest atual

Treino com dados de 1º de janeiro de 2019 a 12 de novembro de 2026 e validação
com origens nas oito semanas finais:

| Subsistema | MAE modelo | MAE semanal | Selecionado | MAPE selecionado |
|---|---:|---:|---|---:|
| Norte | 233,57 | 238,74 | Modelo | 2,39% |
| Nordeste | 581,37 | 539,42 | Baseline semanal | 3,61% |
| Sul | 577,91 | 753,67 | Modelo | 4,30% |
| Sudeste/Centro-Oeste | 2.472,08 | 3.022,29 | Modelo | 5,31% |

Essas métricas são de um único corte temporal e não substituem monitoramento
contínuo. O relatório completo está em `artifacts/models/*.metrics.json`.

### API

Inicie o serviço:

```powershell
ons-api
```

Endpoints:

- `GET /health`: disponibilidade e versão do modelo;
- `GET /v1/forecast`: previsão mais recente para as próximas 168 horas;
- `GET /v1/forecast?origin=2026-11-12T23:00:00`: previsão para uma origem
  histórica existente na gold;
- `GET /docs`: documentação OpenAPI interativa.

Os caminhos podem ser configurados com `ONS_GOLD_PATH` e `ONS_MODEL_PATH`.
Exemplo:

```powershell
$env:ONS_GOLD_PATH = "data/gold/ons_load_hourly.parquet"
$env:ONS_MODEL_PATH = "artifacts/models/latest.joblib"
ons-api
```

### Docker

```powershell
docker build -t power-grid-load-forecasting .
docker run --rm -p 8000:8000 `
  -v "${PWD}/data:/app/data" `
  -v "${PWD}/artifacts:/app/artifacts" `
  power-grid-load-forecasting
```

O container inicia a API. Para executar outro comando, substitua o comando da
imagem, por exemplo `docker run ... power-grid-load-forecasting ons-production`.

### Agendamento

Em produção, execute a rotina depois da segunda atualização diária do ONS. No
Agendador de Tarefas do Windows, use `.venv\Scripts\ons-production.exe`. Em
cron, um exemplo é:

```cron
30 20 * * * /app/.venv/bin/ons-production
```

Os anos devem ser atualizados na virada do calendário. Em um orquestrador,
calcule dinamicamente o ano corrente e o anterior e use o status do processo e
o manifesto como sinais de observabilidade.

## Contratos e validações

A execução falha antes de publicar dados tratados quando encontra:

- coluna obrigatória ausente;
- data pertencente a outro ano;
- subsistema desconhecido ou algum dos quatro subsistemas ausente;
- chave `(timestamp, subsystem)` duplicada;
- instante fora da hora cheia;
- carga nula, não numérica ou negativa;
- recurso anual ausente ou duplicado no catálogo.

O total do SIN só é calculado quando os quatro subsistemas estão presentes no
instante. Caso algum valor falte, `load_sin_mwmed` fica nulo, tornando a falha
visível para as verificações do modelo.

### Esquema silver

| Coluna | Tipo lógico | Descrição |
|---|---|---|
| `timestamp` | datetime | instante informado pelo ONS |
| `subsystem` | string | N, NE, S ou SE |
| `subsystem_name` | string | nome oficial do subsistema |
| `load_mwmed` | float | carga horária em MWmed |
| `year` | int | partição lógica anual |
| `source_resource_id` | string | identificador CKAN |
| `source_sha256` | string | hash do CSV original |
| `ingested_at_utc` | string | instante UTC da ingestão |

### Esquema gold

Uma linha por hora com `timestamp`, `load_n_mwmed`, `load_ne_mwmed`,
`load_s_mwmed`, `load_se_mwmed` e `load_sin_mwmed`.

O ONS publica `din_instante` sem indicação explícita de fuso no arquivo. A
pipeline preserva esse valor. Antes de juntar previsões meteorológicas, deve-se
confirmar a convenção temporal e tratar o histórico de horário de verão.

## Desenvolvimento e testes

```powershell
python -m unittest discover -s tests -v
python -m ruff check src tests
```

Os testes usam dados sintéticos e não dependem da internet. Para um teste de
integração real, execute somente um ano recente em um diretório temporário:

```powershell
ons-load --start-year 2026 --end-year 2026 --data-dir .tmp-data
```

## Operação e recuperação

- Uma falha gera manifesto com `status: failed`; cada arquivo Parquet publicado
  permanece íntegro porque sua troca é atômica.
- Escritas Parquet usam arquivo temporário e troca atômica no mesmo volume.
- Um lock por diretório bloqueia execuções concorrentes.
- Reexecutar o mesmo intervalo é idempotente na silver e na gold.
- CSVs antigos não são apagados: hashes diferentes preservam revisões da fonte.
- Para reconstrução total, guarde a bronze e reexecute o intervalo histórico.
- Monitore idade do último manifesto bem-sucedido, nulos no total SIN,
  quantidade de linhas e variações anormais da carga.

## Limitações e evolução recomendada

- Ainda não há previsão meteorológica, feriados móveis ou eventos especiais.
- O backtest atual usa um corte temporal; a próxima evolução é backtesting com
  múltiplas origens e métricas separadas por faixa do horizonte.
- A API carrega o artefato local. Implantação distribuída deve usar um registro
  de modelos e armazenamento de objetos compartilhado.
- Alertas devem comparar previsão e realizado por subsistema e horizonte,
  detectando degradação antes de promover um novo modelo.
- As revisões do ONS criam risco de viés retrospectivo. Uma avaliação rigorosa
  deve reconstruir cada data usando a versão disponível naquele momento; a
  bronze versionada por hash preserva a matéria-prima para isso.

## Estrutura do código

```text
src/ons_load_pipeline/
├── client.py       # descoberta CKAN, download, retry e timeout
├── config.py       # caminhos e parâmetros operacionais
├── transform.py    # normalização, contrato e camada gold
├── pipeline.py     # orquestração, versionamento e publicação
├── features.py     # atributos temporais sem vazamento
├── training.py     # treino, backtest, seleção e registro
├── forecasting.py  # previsão versionada de 168 horas
├── api.py          # API FastAPI
├── production.py   # atualização diária ponta a ponta
└── cli.py          # interface de ingestão
tests/
├── test_features.py
├── test_pipeline.py
└── test_transform.py
```
