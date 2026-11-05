$ErrorActionPreference = "Stop"

# O script está em src/data; portanto, a raiz do projeto fica dois níveis acima.
$raizProjeto = Resolve-Path (Join-Path $PSScriptRoot "../..")

$pastaZips = Join-Path $raizProjeto "data/inmet/dados_brutos/arquivos_zip"
$pastaExtracao = Join-Path $raizProjeto "data/inmet/dados_brutos/estacoes_automaticas"

$anos = 2015..2025
$resultados = @()

foreach ($ano in $anos) {
    $arquivoZip = Join-Path $pastaZips "$ano.zip"
    $pastaAno = Join-Path $pastaExtracao "$ano"

    if (-not (Test-Path $arquivoZip)) {
        Write-Warning "Arquivo não encontrado: $arquivoZip"

        $resultados += [PSCustomObject]@{
            Ano      = $ano
            Situacao = "ZIP não encontrado"
            CSVs     = 0
        }

        continue
    }

    New-Item -ItemType Directory -Path $pastaAno -Force | Out-Null

    $arquivosExistentes = @(
        Get-ChildItem $pastaAno -Recurse -File -ErrorAction SilentlyContinue
    ).Count

    if ($arquivosExistentes -gt 0) {
        Write-Host "Ano $ano já possui arquivos. Extração ignorada." -ForegroundColor Yellow

        $quantidadeCsv = @(
            Get-ChildItem $pastaAno -Recurse -File -Filter "*.CSV"
        ).Count

        $resultados += [PSCustomObject]@{
            Ano      = $ano
            Situacao = "Já extraído"
            CSVs     = $quantidadeCsv
        }

        continue
    }

    Write-Host "Extraindo dados de $ano..." -ForegroundColor Cyan

    Expand-Archive `
        -LiteralPath $arquivoZip `
        -DestinationPath $pastaAno

    $quantidadeCsv = @(
        Get-ChildItem $pastaAno -Recurse -File -Filter "*.CSV"
    ).Count

    $resultados += [PSCustomObject]@{
        Ano      = $ano
        Situacao = "Extraído"
        CSVs     = $quantidadeCsv
    }
}

Write-Host ""
Write-Host "Resumo da extração:" -ForegroundColor Green

$resultados |
    Format-Table -AutoSize