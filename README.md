# adcrs

Local audio synchronization tool.

The MVP finds a fixed audio offset using spectral features and energy patterns, then muxes the synchronized audio into a target video while preserving the video stream whenever possible.

## Requirements

- Python 3.10+
- FFmpeg
- ffprobe
- NumPy

On Fedora/Nobara:

```bash
sudo dnf install ffmpeg
```

## Installation

From the project directory:

```bash
python3 -m pip install -e .
```

This installs the `audiosync` and `audiosync-web` commands.

## CLI

Analyze and create the output file:

```bash
audiosync target.mkv source.mkv
```

Default output:

```text
target.dubbed.mkv
```

Analyze without creating an output file:

```bash
audiosync target.mkv source.mkv --dry-run
```

Choose an output path and export reports:

```bash
audiosync target.mkv source.mkv \
  --output synchronized.mkv \
  --report sync.json \
  --csv sync.csv
```

If confidence is below the configured threshold, the command returns `MANUAL REVIEW` and does not create the output file.

## Web UI

Start the local server:

```bash
audiosync-web
```

Open `http://127.0.0.1:8765` in a browser. Without installing the package:

```bash
python3 -m audiosync.web
```

The UI lets you:

- select the target video and source audio/video;
- analyze the alignment;
- inspect offset, tempo correction, confidence, matches, and segments;
- create and download the final MKV.

Files are processed locally by the server. Nothing is uploaded to an external service.

## Tests

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q audiosync
```

## How it works

1. `ffprobe` validates both files and locates their audio streams.
2. `ffmpeg` extracts mono 8 kHz PCM audio for analysis.
3. The audio is converted into spectral features and energy flux.
4. Distributed target windows are searched in the source audio using normalized correlation.
5. Matches are used to estimate the offset, a preliminary time ratio, and confidence.
6. The source audio is muxed into the target video with `-c:v copy`.

## Current limitations

- The MVP primarily estimates a fixed offset.
- Progressive drift is not corrected reliably yet.
- Cuts, missing intros, recaps, logos, and multiple offset changes are not segmented yet.
- Batch episode processing is not implemented.
- The analysis does not yet use L-R separation or special center-channel handling.
- The output audio is re-encoded to AAC when muxing is required.
- The Web UI is intentionally local and simple, without authentication. Do not expose it publicly.

## Structure

```text
audiosync/
  cli.py       # CLI and muxing
  ffmpeg.py    # ffprobe, validation, and extraction
  sync.py      # analysis and reports
  web.py       # local Web UI server
  web/
    index.html

tests/
  test_sync.py
```

The project is being developed in phases. The next natural step is improving drift estimation before automating cuts and segmented timelines.
