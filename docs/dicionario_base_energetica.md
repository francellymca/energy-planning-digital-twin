# Dicionário da base energética mensal

## Escopo

A base integrada reúne o consumo mensal de energia elétrica da EPE por unidade
da federação e a carga de energia do ONS por subsistema. O recorte compreende
janeiro de 2015 a dezembro de 2025 para os estados de São Paulo, Minas Gerais,
Rio de Janeiro, Paraná, Bahia e Rio Grande do Sul.

A unidade de observação é uma combinação de mês e unidade da federação. O
consumo da EPE constitui a variável energética principal. A carga do ONS é uma
variável operacional complementar, repetida para os estados pertencentes ao
mesmo subsistema. As duas medidas não são tratadas como equivalentes.

## Campos

| Campo | Tipo | Unidade | Fonte | Definição |
|---|---|---|---|---|
| `mes` | data | mês | EPE e ONS | Primeiro dia do mês de referência, usado como chave temporal. |
| `uf` | texto | código da UF | EPE | Unidade da federação analisada: SP, MG, RJ, PR, BA ou RS. |
| `regiao` | texto | categoria | EPE | Região geográfica brasileira associada à UF. |
| `sistema_epe` | texto | categoria | EPE | Sistema elétrico informado pela EPE para a UF. |
| `consumo_mwh` | número decimal | MWh | EPE | Consumo mensal de energia elétrica, somado entre classes de consumo e ambientes de contratação. |
| `numero_consumidores` | número inteiro | unidades consumidoras | EPE | Total mensal de unidades consumidoras, somado entre classes de consumo e ambientes de contratação. |
| `id_subsistema` | texto | código | EPE e ONS | Código harmonizado do subsistema: N, NE, S ou SE. |
| `consumo_mwh_por_consumidor` | número decimal | MWh por unidade consumidora | derivada | Razão entre `consumo_mwh` e `numero_consumidores`. |
| `nom_subsistema` | texto | categoria | ONS | Nome do subsistema associado ao código do ONS. |
| `carga_mwmed_media_mes` | número decimal | MWmed | ONS | Média mensal das cargas médias diárias do subsistema. |
| `energia_estimada_mwh_mes` | número decimal | MWh | derivada do ONS | Soma mensal da carga média diária multiplicada por 24 horas. É uma aproximação energética da carga do subsistema. |
| `dias_no_mes` | número inteiro | dias | derivada | Quantidade de datas diárias do ONS utilizadas na agregação mensal. |
| `valores_imputados` | número inteiro | registros | derivada | Quantidade de valores diários do ONS interpolados no mês e subsistema. |
| `papel_epe` | texto | categoria | metodologia | Identifica a EPE como fonte da variável principal de consumo estadual. |
| `papel_ons` | texto | categoria | metodologia | Identifica o ONS como fonte da variável complementar de carga por subsistema. |

## Definições das fontes

### Consumo mensal da EPE

Segundo o dicionário da EPE, `Consumo` representa o consumo de energia elétrica
em MWh. A tabela por UF é agregada por unidade da federação, região,
subsistema, classe de consumo e ambiente de contratação.

As classes disponíveis são residencial, industrial, comercial, rural e outros.
O ambiente de contratação distingue consumidores cativos e livres.

### Número de consumidores da EPE

A EPE define uma unidade consumidora como o conjunto de instalações e
equipamentos elétricos caracterizado pelo recebimento de energia elétrica em um
único ponto de entrega, com medição individualizada e correspondente a um único
consumidor.

A soma do número de consumidores entre UF, classe e ambiente de contratação foi
validada contra a tabela regional agregada da própria EPE. Foram reconciliadas
9.974 combinações de mês, região, sistema, classe e tipo de consumidor entre
2015 e 2025, sem diferenças no número de consumidores.

### Carga de energia do ONS

A carga do ONS está disponível em MWmed por dia e subsistema. Para gerar a
aproximação mensal em MWh, cada valor diário é multiplicado por 24 horas e os
resultados são somados dentro do mês.

A carga por subsistema abrange perdas e outros componentes do atendimento do
Sistema Interligado Nacional. Por isso, ela não deve ser interpretada como o
consumo estadual medido pela EPE.

## Tratamento de valores ausentes

Os arquivos anuais do ONS possuem cobertura completa entre 2015 e 2025. Foram
identificados quatro valores ausentes em 9 de abril de 2015, um para cada
subsistema. Esses valores foram estimados por interpolação linear entre 8 e 10
de abril de 2015. A intervenção é registrada no campo `valores_imputados`.

Não foram identificadas ausências na série mensal de consumo da EPE para os
seis estados e o período selecionado.

## Regras de uso

- Usar `consumo_mwh` como variável-alvo para previsão estadual.
- Usar as medidas do ONS apenas como variáveis operacionais complementares.
- Não interpretar diferenças entre EPE e ONS como erro sem considerar as
  diferenças conceituais, espaciais e metodológicas das fontes.
- Preservar os arquivos originais e gerar novamente a base processada por meio
  do script `src/data/process_ons_epe.py`.
