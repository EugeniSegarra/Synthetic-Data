# Synthetic Data Generator

This project generates synthetic OHLC price series using GARCH models and a Student t-copula. It downloads historical data from Yahoo Finance, fits univariate GARCH models, and simulates correlated returns to produce new price paths.

## Requisitos

- Python 3.11
- Las dependencias están listadas en `requirements.txt`.

## Uso

```bash
pip install -r requirements.txt
python synth_data.py --start-date 2018-01-01 --tickers 5
```

Los argumentos permiten especificar la fecha inicial, fecha final opcional y el número de activos del S&P 500 a usar. También se puede definir un archivo de log y el nivel de logging.

## Tests

Ejecute `pytest` en la raíz del repositorio para lanzar las pruebas unitarias.
