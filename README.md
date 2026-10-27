# Timestral Telemetry & Forecast Workbench (Webapp + AI Agent API)

Piattaforma web ed endpoint API ad alte prestazioni per **Mistral Timestral** (`mistralai/Timestral-5M`, `28M`, `123M`, `209M`), progettati per:
- Previsioni probabilistiche zero-shot su serie temporali generiche con riscontro empirico (*holdout backtesting*).
- **Crypto Trading Workbench** dedicato (`/crypto`) per segnali operativi in tempo reale (BUY/SELL/HOLD), backtest walk-forward con separazione In-Sample / Out-of-Sample e simulatore di investimento manuale.
- Integrazione nativa per **Agenti AI** tramite OpenAI/Mistral Function Calling, MCP (Model Context Protocol), OpenAPI 3.1 e `llms.txt`.
- Ottimizzazione per **Raspberry Pi 5** con single-slot memory management, monitoraggio telemetrico hardware e auto-eviction dopo 300s di inattività.

---

## Fonti Ufficiali Mistral AI
- **Repository Ufficiale Timestral**: [https://github.com/mistralai/timestral/](https://github.com/mistralai/timestral/)
- **Annuncio & Documentazione**: [https://mistral.ai/news/timestral](https://mistral.ai/news/timestral)
- **Collezione Modelli Hugging Face**: [https://huggingface.co/collections/mistralai/timestral](https://huggingface.co/collections/mistralai/timestral)

---

## Funzionalità & Modalità Native

### 1. General Forecasting & Advanced Variants
Timestral emette **21 quantili non incrociati** (`0.01, 0.05, 0.10, ..., 0.50, ..., 0.90, 0.95, 0.99`) fino a `32.768` step di contesto temporale:
1. **Zero-Shot Probabilistic Forecasting (`POST /api/v1/forecast`)**: Previsione multi-step con mediana (`q50`) e intervalli di confidenza (80%, 90%, 98%).
2. **Holdout Benchmark su Serie Reali (`POST /api/v1/backtest`)**: Valutazione rigorosa su serie orarie e giornaliere con confronto rispetto a baseline *Seasonal Naive* (metriche MAE, RMSE, sMAPE, Pinball Loss, Coverage 80%/90%).
3. **Multi-Model Comparison (`POST /api/v1/backtest/compare-models`)**: Confronto multi-taglia `5M` vs `28M` vs `123M` vs `209M` su latenza, RAM e precisione.
4. **Zero-Shot Anomaly Detection (`POST /api/v1/anomalies`)**: Rilevamento anomalie e calcolo *z-quantile score* sui punti che perforano i quantili estremi.
5. **Simulazione What-If con Covariate Future (`POST /api/v1/scenario`)**: Valutazione dell'impatto di eventi pianificati sfruttando `ROLE_KNOWN_COV`.
6. **Tail-Risk, VaR & Probabilità di Superamento (`POST /api/v1/risk`)**: Calcolo di `VaR_95`, `VaR_99` e probabilità esatta $P(Y_t > \tau)$ tramite inversione CDF quantile.
7. **Stress-Test Dati Mancanti / Sensor Dropout (`POST /api/v1/missing-stress`)**: Resilienza a drop causali e buchi `NaN` con `observation_mask`.
8. **Previsione Multivariata Congiunta (`POST /api/v1/multivariate`)**: Previsione simultanea di più serie correlate tramite cross-variate attention.

### 2. Crypto Trading Workbench (`/crypto`)
Area dedicata al trading quantitativo su **BTC/USDT** ed **ETH/USDT** (timeframe 15m, 1h, 4h, 1d):
- **Segnali Live**: BUY / SELL / HOLD calcolati in tempo reale sulla distribuzione a 21 quantili di Timestral.
- **Walk-Forward Backtesting**: Separazione rigida tra **In-Sample (70%)** e **Out-of-Sample (30% Cieco)** per prevenire l'overfitting.
- **Metriche Finanziarie Certificate**: **Profit Factor** (gross profit / gross loss, $>1.5$ ottimale), **Win Rate %**, **Max Drawdown %**, **Sharpe Ratio** annualizzato e commissioni reali (0.075% taker fee).
- **Curva Equity Continua**: Calcolo mark-to-market candela per candela con linea di separazione IS vs OOS.
- **Simulatore Investimento Manuale (`POST /api/v1/crypto/simulate-investment`)**: Calcolo in tempo reale di resa stimata base ($q50$), target bull ($q90$), rischio bear ($q10$), probabilità di profitto e allerta liquidazione per posizioni con leva (1x, 2x, 3x, 5x).

---

## Discovery Directory per Agenti AI e Sviluppatori

Tutti gli endpoint di discovery sono consultabili a costo zero senza caricare pesi in memoria RAM:

| Endpoint | Formato | Audience | Descrizione |
|---|---|---|---|
| `GET /health` | JSON | AI Agent & Umani | Stato modello, memoria RAM, CPU temp e sitemap completa |
| `GET /docs` | HTML | Umani & Dev | Swagger UI interattiva (OpenAPI 3.1) |
| `GET /doc` | JSON | AI Agent & Umani | Documentazione rapida per agenti |
| `GET /redoc` | HTML | Umani & Dev | Documentazione ReDoc strutturata |
| `GET /openapi.json` | JSON | AI Agent / Tools | Specifica OpenAPI 3.1 completa |
| `GET /readme` | Markdown | AI Agent & Umani | Questo file in formato Markdown |
| `GET /llms.txt` | Text | AI Agent / LLM | Manifesto di discovery standard per LLM |
| `GET /api/v1/agent-schema` | JSON | AI Agent | Tool array pronti per OpenAI / Mistral Function Calling e MCP |

---

## Installazione & Esecuzione

### 1. Prerequisiti & Dipendenze
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Avvio Locale del Server
```bash
.venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8765
```
Dashboard aperte:
- **Interfaccia Generale & Benchmark**: `http://localhost:8765/`
- **Crypto Trading Workbench**: `http://localhost:8765/crypto`
- **Swagger UI**: `http://localhost:8765/docs`

### 3. Esecuzione Test Suite (Ponytail verification)
```bash
.venv/bin/python test_app.py
```

### 4. Servizio Systemd su Linux / Raspberry Pi
Un file di servizio pronto per l'esecuzione continua è disponibile in `/etc/systemd/system/timestral.service`:
```bash
sudo systemctl daemon-reload
sudo systemctl enable timestral.service
sudo systemctl start timestral.service
sudo systemctl status timestral.service
```

---

## Licenza
Apache 2.0
