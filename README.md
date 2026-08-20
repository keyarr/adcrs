# adcrs

Sincronizador local de áudio dublado com vídeo de melhor qualidade.

O MVP encontra um offset fixo usando características espectrais e energia do áudio. Depois gera um novo arquivo preservando o vídeo sem recodificação sempre que possível.

## Requisitos

- Python 3.10+
- FFmpeg
- ffprobe
- NumPy

No Fedora/Nobara:

```bash
sudo dnf install ffmpeg
```

## Instalação

No diretório do projeto:

```bash
python3 -m pip install -e .
```

Isso instala os comandos `audiosync` e `audiosync-web`.

## CLI

Analisar e gerar o arquivo final:

```bash
audiosync video_hq.mkv video_dublado_sd.mkv
```

Saída padrão:

```text
video_hq.dubbed.mkv
```

Fazer apenas a análise:

```bash
audiosync video_hq.mkv video_dublado_sd.mkv --dry-run
```

Escolher saída e exportar relatórios:

```bash
audiosync video_hq.mkv video_dublado_sd.mkv \
  --output video_hq.dubbed.mkv \
  --report sync.json \
  --csv sync.csv
```

Se a confiança ficar abaixo do limite, o programa retorna `MANUAL REVIEW` e não cria o arquivo final.

## Web UI

Inicie o servidor local:

```bash
audiosync-web
```

Abra `http://127.0.0.1:8765` no navegador. Também funciona sem instalação do pacote:

```bash
python3 -m audiosync.web
```

A interface permite:

- selecionar o vídeo target e o arquivo dublado;
- analisar o alinhamento;
- visualizar offset, correção temporal, confiança, matches e segmentos;
- criar o MKV final e baixar o resultado.

Os arquivos enviados são processados localmente pelo servidor. Não existe upload para serviço externo.

## Testes

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q audiosync
```

## Como funciona

1. `ffprobe` valida os arquivos e encontra as trilhas de áudio.
2. `ffmpeg` extrai uma cópia mono PCM em 8 kHz para análise.
3. O áudio vira uma sequência de características espectrais e fluxo de energia.
4. Janelas distribuídas do target são procuradas no áudio source por correlação normalizada.
5. Os pontos encontrados estimam offset, relação temporal preliminar e confiança.
6. O áudio dublado é muxado no vídeo target com `-c:v copy`.

## Limitações atuais

- O MVP estima principalmente offset fixo.
- Drift progressivo ainda não é corrigido de forma confiável.
- Cortes, logos ausentes, recaps e múltiplas mudanças de offset ainda não são segmentados.
- Batch de episódios ainda não está implementado.
- A análise não usa L-R nem tratamento específico para canal central.
- O áudio final é recodificado para AAC quando o mux é necessário.
- A Web UI é deliberadamente local e simples, sem autenticação. Não exponha a porta publicamente.

## Estrutura

```text
audiosync/
  cli.py       # CLI e mux
  ffmpeg.py    # ffprobe, validação e extração
  sync.py      # análise e relatório
  web.py       # servidor local da Web UI
  web/
    index.html

tests/
  test_sync.py
```

Projeto em fases. O próximo passo natural é melhorar a estimativa de drift antes de tentar automatizar cortes e timelines segmentadas.
