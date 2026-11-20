---
title: California Beach Enterococcus Risk
sdk: docker
app_port: 7860
---

# California Beach Enterococcus Risk

FastAPI app: open **`/docs`** for interactive API documentation.
`GET /health` reports whether the model and history are ready; `GET /predict`
accepts a beach ID and an ISO date.

This source upload does not contain a trained model or a station history snapshot.
Until a verified serving export is installed at `/home/user/app/artifacts`,
`/health` and `/predict` return **503**. Automatic Hub artifact download is pending.
The documentation page can still load while prediction is unavailable.

The local project demo uses synthetic observations. Its probabilities are not
real California water-quality estimates or official beach advisories.
