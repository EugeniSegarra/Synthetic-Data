#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import mplfinance as mpf

from arch import arch_model
from datetime import timedelta
from scipy import stats

# Para la t-copula manual
from scipy.stats import t as tdist
from scipy.stats import chi2
from scipy.stats import skewnorm

# Para la validación de colas
import statsmodels.api as sm  # para QQ-plot más avanzado (opcional)

##############################################################################
# LOGGING GLOBAL
##############################################################################
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

##############################################################################
# A. UTILS: near-PD (clipping y Higham), rank-transform
##############################################################################
def project_psd(A):
    """
    Proyecta la matriz A en el espacio de matrices PSD (semidefinidas positivas)
    tomando los autovalores negativos y llevándolos a 0, y re-simetrizando.
    """
    # Autovalores y vectores
    vals, vecs = np.linalg.eigh(A)
    vals[vals < 0] = 0  # clip en 0
    return vecs @ np.diag(vals) @ vecs.T

def higham_nearPD(A, max_iter=50, tol=1e-7):
    """
    Implementación del algoritmo de Higham para proyectar una matriz
    simétrica en la más cercana PSD (en norma Frobenius).
    """
    n = A.shape[0]
    Y = A.copy()
    normA = np.linalg.norm(A, 'fro')
    for k in range(max_iter):
        # Paso 1: proyección en S (matrices simétricas)
        X = 0.5 * (Y + Y.T)
        # Paso 2: proyección en PSD
        R = project_psd(X)
        # Corrección
        diff = R - X
        # Paso 3: Centrar alrededor de A
        Y = Y + diff
        # Asegurar simetría
        Y = 0.5 * (Y + Y.T)
        # Criterio de convergencia
        if np.linalg.norm(diff, 'fro') / normA < tol:
            break

    return Y

def ensure_positive_definite(
    corr_matrix,
    eps=1e-6,
    use_higham=False
):
    """
    Ajusta una matriz corr_matrix para que sea semidefinida positiva.
    Dos métodos:
      - Clipping de autovalores (rápido, approach clásico).
      - Higham (iterativo, más exacto).
    """
    if use_higham:
        # Usa Higham
        R0 = higham_nearPD(corr_matrix)
        # Normaliza la diagonal a 1 si es correlación
        diag_sqrt = np.sqrt(np.diag(R0))
        R0 /= np.outer(diag_sqrt, diag_sqrt)
        return R0
    else:
        # Clipping directo
        R = corr_matrix.copy()
        vals, vecs = np.linalg.eigh(R)
        vals_clipped = np.clip(vals, eps, None)
        R_pdc = (vecs @ np.diag(vals_clipped) @ vecs.T)

        # Re-normalizamos la diagonal a 1 (si es correlación).
        diag_sqrt = np.sqrt(np.diag(R_pdc))
        R_pdc /= np.outer(diag_sqrt, diag_sqrt)

        return R_pdc

def rank_transform(x):
    """
    Convierte un vector 1D x en una pseudo-variable Normal(0,1)
    usando 'rankdata' y ppf normal.
    """
    ranks = stats.rankdata(x)
    # Evitar 0 y 1 exactos
    N = len(x)
    probs = (ranks - 0.5) / N
    return stats.norm.ppf(probs)

##############################################################################
# B. IMPLEMENTACIÓN UNIVARIANTE: StudentTUnivariateManual
##############################################################################
class StudentTUnivariateManual:
    """
    Ajusta un t-Student univariante a datos 1D usando método MLE con scipy,
    para obtener loc, scale, df. Reemplaza la necesidad de 'copulas.univariate'.
    """

    def __init__(self):
        self.loc_ = None
        self.scale_ = None
        self.df_ = None

    def fit(self, data):
        """
        Ajustar distribución t-Student a 'data' (numpy array 1D),
        usando MLE con scipy.
        """
        def negloglike(params, x):
            df, loc, scale = params
            if scale <= 0 or df <= 2:
                return np.inf
            return -np.sum(tdist.logpdf(x, df, loc=loc, scale=scale))

        loc_init = np.mean(data)
        scale_init = np.std(data, ddof=1)
        df_init = 8.0  # un guess

        from scipy.optimize import minimize
        res_opt = minimize(
            negloglike,
            x0=[df_init, loc_init, scale_init],
            args=(data,),
            bounds=[(2.01, 500.0), (-np.inf, np.inf), (1e-9, np.inf)],
            method='L-BFGS-B'
        )
        df_mle, loc_mle, scale_mle = res_opt.x
        self.df_ = df_mle
        self.loc_ = loc_mle
        self.scale_ = scale_mle

    def cdf(self, x):
        """CDF de la t-Student con df, loc, scale en x."""
        return tdist.cdf(x, df=self.df_, loc=self.loc_, scale=self.scale_)

    def ppf(self, q):
        """PPF (inverse CDF) de la t-Student con df, loc, scale en q."""
        return tdist.ppf(q, df=self.df_, loc=self.loc_, scale=self.scale_)

##############################################################################
# C. COPULA MULTIVARIANTE T (ManualStudentTCopula) CON MEJORAS
##############################################################################
class ManualStudentTCopula:
    """
    Implementa una t-cópula real (elíptica) con 'nu' global compartido.
    - Ajusta un StudentTUnivariateManual a cada dimensión (marginales).
    - Determina un nu_global con una regla "soft" si no se especifica.
    - Opción para hacer rank transform previo al cálculo de la correlación
      (para capturar mejor la dependencia de colas).
    - Permite usar near-PD (clipping) o Higham para garantizar la PSD.
    """

    def __init__(
        self,
        nu_global=None,
        use_rank_transform=False,
        use_higham_pd=False
    ):
        """
        :param nu_global: None => se auto-calcula (con la regla "soft")
        :param use_rank_transform: bool => si True, se hace rank_transform
        :param use_higham_pd: bool => si True, se usa Higham para near-PD
        """
        self.n_dim = None
        self.marginals = []
        self.R = None
        self.chol = None
        self.nu_global = nu_global
        self.use_rank_transform = use_rank_transform
        self.use_higham_pd = use_higham_pd

    def fit(self, data):
        """
        data: numpy array (n_samples, n_dim)
        """
        n_samples, n_dim = data.shape
        self.n_dim = n_dim

        # 1) Fit univariate t a cada columna
        self.marginals = []
        nus = []
        for i in range(n_dim):
            dist_t = StudentTUnivariateManual()
            dist_t.fit(data[:, i])  # Ajusta la marginal
            self.marginals.append(dist_t)
            nus.append(dist_t.df_)

        # 2) Determinar nu_global (soft approach)
        if self.nu_global is None:
            med_nu = np.median(nus)
            if med_nu > 4:
                # Reducción suave
                self.nu_global = max(med_nu * 0.8, 2.0)
            else:
                self.nu_global = med_nu
            logger.info(f"[t-Copula] nu_global calculado ~ {self.nu_global:.2f} con soft approach.")
        else:
            logger.info(f"[t-Copula] nu_global definido por usuario = {self.nu_global:.2f}.")

        # 3) Transformar a U_i = F_i(data_i)
        U = np.zeros_like(data)
        for i in range(n_dim):
            U[:, i] = self.marginals[i].cdf(data[:, i])

        # 4) Dependiendo del flag, transformamos U => Y
        #    a) sin rank_transform => Y = t.ppf(U, df=nu_global)
        #    b) con rank_transform => Y = rank_transform de data (NO de U)
        #       (porque rank_transform se hace normalmente con la data real)
        #       Pero para no romper la semántica, lo aplicamos en el espacio U
        #       Se puede discutir. Aquí lo hago con la X real para un
        #       "Normal Score Transform" más puro.
        if self.use_rank_transform:
            # rank-transform por columnas
            Y = np.zeros_like(data)
            for i in range(n_dim):
                Y[:, i] = rank_transform(data[:, i])
            logger.info("[t-Copula] Se ha aplicado rank transform antes de la correlación.")
        else:
            Y = tdist.ppf(U, df=self.nu_global)

        # 5) Calcular la correlación empírica en Y
        R = np.corrcoef(Y, rowvar=False)
        R_pd = ensure_positive_definite(R, use_higham=self.use_higham_pd)
        self.R = R_pd

        # Factor de Cholesky
        self.chol = np.linalg.cholesky(self.R)
        logger.info("[t-Copula] Ajuste completado (matriz correl near-PD).")

    def sample(self, n_samples):
        """
        Generar muestras (n_samples, n_dim) a partir de la t-copula ajustada.
        Pasos:
          1) X ~ t_{\nu_global}(0, R).
             - Equivale a Z ~ N(0,R), W ~ Chi2(nu)/nu, X = Z / sqrt(W/nu).
          2) U = cdf_t_\nu(X)
          3) x_i = ppf_i(U_i) con la marginal i
        """
        if self.n_dim is None:
            raise ValueError("Debes ejecutar .fit(data) antes de .sample().")

        n_dim = self.n_dim
        # a) Z ~ N(0, R)
        Z = np.random.normal(size=(n_samples, n_dim)) @ self.chol.T
        # b) W ~ Chi2(nu_global) / nu_global
        W = chi2.rvs(self.nu_global, size=n_samples) / self.nu_global
        W = np.sqrt(W)

        # c) X = Z / sqrt(W)
        X = Z / W[:, None]

        # 2) U_i = cdf_t_\nu_global(X_i)
        U = tdist.cdf(X, df=self.nu_global)

        # 3) x_i = ppf_i(U_i)
        samples = np.zeros((n_samples, n_dim))
        for i in range(n_dim):
            samples[:, i] = self.marginals[i].ppf(U[:, i])

        return samples

##############################################################################
# D. VALIDACIÓN DE LA T-COPULA
##############################################################################
def validate_t_copula(copula_obj, data_real, n_samples=10000):
    """
    Genera muestras sintéticas a partir de la copula
    y compara la matriz de correlación de Spearman
    con la de los datos reales.
    """
    # Real spearman
    real_spearman = pd.DataFrame(data_real).corr(method='spearman')
    # Synthetic
    sim_data = copula_obj.sample(n_samples)
    sim_spearman = pd.DataFrame(sim_data).corr(method='spearman')

    logger.info("==== Spearman Correlation (Real vs. Synthetic) ====")
    logger.info("Real Spearman:\n" + real_spearman.round(3).to_string())
    logger.info("Synthetic Spearman:\n" + sim_spearman.round(3).to_string())
    return real_spearman, sim_spearman

##############################################################################
# E. DESCARGA DE TICKERS S&P500
##############################################################################
def get_sp500_symbols():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    df = tables[0]
    tickers = df['Symbol'].tolist()
    tickers = [t.replace('.', '-') for t in tickers]
    return tickers

##############################################################################
# F. DESCARGA DE PRECIOS DESDE YFINANCE
##############################################################################
def download_prices(tickers, start_date="2018-01-01", end_date=None):
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Adj Close']
    if isinstance(data, pd.Series):
        data = data.to_frame()
    data = data.dropna(how='all')
    if len(tickers) == 1 and data.shape[1] == 1:
        data.columns = tickers
    return data

##############################################################################
# G. AJUSTE GARCH CON DRIFT Y T-STUDENT, + ESCALADO
##############################################################################
def fit_best_garch(returns_series, dist='StudentsT', max_p=2, max_q=2, mean='Constant'):
    """
    Ajusta un GARCH con búsqueda en p,q en [1..max_p], [1..max_q]
    y elige el que tenga menor BIC.
    """
    best_model = None
    best_bic = np.inf

    for p in range(1, max_p + 1):
        for q in range(1, max_q + 1):
            try:
                am = arch_model(returns_series, p=p, q=q,
                                dist=dist, mean=mean, rescale=False)
                res = am.fit(disp="off")
                if res.bic < best_bic:
                    best_bic = res.bic
                    best_model = res
            except:
                continue

    return best_model

def fit_garch_t(df_prices):
    """
    1) Calcula log-returns.
    2) Ajusta GARCH t-Student para cada activo por separado
       con escalado de la serie (para uniformizar magnitudes).
    3) Devuelve:
       - "models": dict con los resultados de arch_model.fit() final
       - "residuals": DataFrame de residuos estandarizados
       - "scales": dict con el factor de escalado usado
       - "daily_vol": dict con la desviación std real de cada activo
    """
    log_returns = np.log(df_prices).diff().dropna()
    fitted_models = {}
    std_resids = pd.DataFrame(index=log_returns.index, columns=log_returns.columns)
    scale_dict = {}
    daily_vol_dict = {}

    for col in log_returns.columns:
        series_raw = log_returns[col].dropna()
        vol_est = series_raw.std()
        if vol_est <= 0:
            continue

        # Escalamos la serie => series_scaled = series_raw / scale_factor
        scale_factor = vol_est / 1.0
        series_scaled = series_raw / scale_factor

        res = fit_best_garch(series_scaled, dist='StudentsT', max_p=2, max_q=2, mean='Constant')
        if res is None:
            continue

        cond_vol = res.conditional_volatility
        raw_resids = res.resid
        z_t_scaled = raw_resids / cond_vol  # std resid en la escala "scaled"

        fitted_models[col] = res
        std_resids[col] = z_t_scaled.reindex(std_resids.index, fill_value=np.nan)
        scale_dict[col] = scale_factor
        daily_vol_dict[col] = vol_est

        # Validación univariante: QQ-plot contra t-Student con el df estimado
        df_model = res.params.get('nu', 8.0)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        sm.qqplot(z_t_scaled.dropna(), dist=stats.t, distargs=(df_model,), line='45', ax=ax)
        ax.set_title(f"QQ-Plot Residuals - {col} (t-dist, df~{df_model:.2f})")
        plt.show()

    std_resids = std_resids.dropna()
    return {
        "models": fitted_models,
        "residuals": std_resids,
        "scales": scale_dict,
        "daily_vol": daily_vol_dict
    }

##############################################################################
# H. CONSTRUCCIÓN T-COPULA Y VALIDACIÓN
##############################################################################
def build_t_copula(
    std_resids,
    nu_global=None,
    use_rank_transform=False,
    use_higham_pd=False
):
    data = std_resids.values  # shape (n_samples, n_assets)
    copula = ManualStudentTCopula(
        nu_global=nu_global,
        use_rank_transform=use_rank_transform,
        use_higham_pd=use_higham_pd
    )
    copula.fit(data)
    return copula

def simulate_correlated_stdresids_tcopula(copula_obj, n_samples):
    return copula_obj.sample(n_samples)

##############################################################################
# I. RECONSTRUCCIÓN DE PRECIOS MEDIANTE GARCH
##############################################################################
def reconstruct_garch_paths(sim_std_resids, fitted_garch_dict, start_prices):
    """
    sim_std_resids: (n_days, n_assets) simulados
    fitted_garch_dict: dict con keys "models", "scales"
    start_prices: array con el precio final real
    """
    models = fitted_garch_dict["models"]
    scale_dict = fitted_garch_dict["scales"]

    asset_names = list(models.keys())
    n_days, n_assets = sim_std_resids.shape
    if len(asset_names) != n_assets:
        raise ValueError("Dimensión inconsistente entre sim_std_resids y 'models'.")

    simulated_prices = np.zeros((n_days, n_assets))
    simulated_prices[0, :] = start_prices

    cond_var_prev = np.zeros(n_assets)
    resid_prev = np.zeros(n_assets)

    # Inicializamos con el último valor cond_vol y resid
    for j, asset in enumerate(asset_names):
        arch_res = models[asset]
        last_cond_vol = arch_res.conditional_volatility.iloc[-1]
        cond_var_prev[j] = last_cond_vol**2
        resid_prev[j] = arch_res.resid.iloc[-1]

    # Iteramos día a día
    for t in range(1, n_days):
        for j, asset in enumerate(asset_names):
            arch_res = models[asset]
            p = arch_res.params
            mu = p.get('mu', 0.0)

            # Manejo GARCH(p,q)
            alpha_keys = [k for k in p.keys() if 'alpha' in k]
            beta_keys = [k for k in p.keys() if 'beta' in k]

            omega = p['omega']
            res_sq = resid_prev[j]**2
            cond_var_t = omega

            # Sumatorio de las alphas
            for a_k in alpha_keys:
                cond_var_t += p[a_k] * res_sq

            # Sumatorio de las betas
            cond_var_t += sum(p[b_k] * cond_var_prev[j] for b_k in beta_keys)

            cond_vol_t = np.sqrt(cond_var_t)
            z_t = sim_std_resids[t, j]

            # En la escala "scaled", resid = cond_vol_t * z_t
            r_t_rescaled = cond_vol_t * z_t

            # Des-escalar: r_t_original = (r_t_rescaled + mu) * scale_factor
            scale_factor = scale_dict[asset]
            r_t_original = (r_t_rescaled + mu) * scale_factor

            # Reconstruimos el precio (log-ret)
            simulated_prices[t, j] = simulated_prices[t-1, j] * np.exp(r_t_original)

            # Actualizamos estados
            resid_prev[j] = r_t_rescaled
            cond_var_prev[j] = cond_var_t

    df_sim_close = pd.DataFrame(simulated_prices, columns=asset_names)
    return df_sim_close

##############################################################################
# J. GENERAR OHLC A PARTIR DE CLOSE (CON PATRÓN INTRADÍA MEJORADO)
##############################################################################
def generate_ohlc_from_close(
    df_close,
    daily_vol_dict,
    intraday_vol_factor=1.0,
    seed=42,
    use_intraday_pattern=True
):
    """
    Genera OHLC con un modelo simple pero algo más realista:
    - Gaps diarios con skewnorm
    - Rango intradía escalado por daily_vol y un factor sinusoidal (opcional)
    """
    np.random.seed(seed)
    n_days, n_assets = df_close.shape
    asset_names = df_close.columns
    all_ohlc = []

    for asset in asset_names:
        close_vals = df_close[asset].values
        arr_open = np.zeros(n_days)
        arr_high = np.zeros(n_days)
        arr_low = np.zeros(n_days)
        arr_close = close_vals.copy()

        arr_open[0] = close_vals[0]
        base_gap_sigma = 0.01 * daily_vol_dict[asset]

        for t in range(1, n_days):
            # Gap con skewnorm
            gap = skewnorm.rvs(a=2, loc=0, scale=base_gap_sigma)
            # Limitamos gap para evitar extremos
            gap = np.clip(gap, -0.05, 0.05)
            arr_open[t] = close_vals[t-1] * (1 + gap)

        # Rango intradía
        for t in range(n_days):
            # Podemos escalar la volatilidad intradía según un patrón
            if use_intraday_pattern:
                # por ejemplo, una oscilación sinusoidal con periodo 24
                # (asumiendo "t" ~ horas, a modo figurado)
                factor_t = 1.0 + 0.5 * np.sin(2 * np.pi * t / 24.0)
            else:
                factor_t = 1.0

            daily_range = intraday_vol_factor * abs(
                np.random.normal(0, base_gap_sigma)
            ) * factor_t

            mn = min(arr_open[t], arr_close[t])
            mx = max(arr_open[t], arr_close[t])

            x = np.random.beta(2, 2)
            up_rng = daily_range * x
            dn_rng = daily_range * (1 - x)

            arr_high[t] = max(mx + up_rng, mx)
            arr_low[t] = min(mn - dn_rng, mn)

        df_a = pd.DataFrame({
            "Open": arr_open,
            "High": arr_high,
            "Low": arr_low,
            "Close": arr_close
        }, index=df_close.index)
        df_a["Asset"] = asset
        all_ohlc.append(df_a)

    df_ohlc = pd.concat(all_ohlc, axis=0)
    df_ohlc.index.name = "Date"
    df_ohlc.reset_index(inplace=True)
    df_ohlc.set_index(["Date", "Asset"], inplace=True)
    return df_ohlc

##############################################################################
# K. GRAFICADO
##############################################################################
def plot_ohlc_candlestick(df_ohlc, asset_name):
    df_a = df_ohlc.xs(asset_name, level="Asset").copy()
    df_a.index = pd.DatetimeIndex(df_a.index)
    df_a.sort_index(inplace=True)
    df_a["Volume"] = 0
    mpf.plot(df_a, type='candle', style='charles',
             title=f"Candlestick - {asset_name}",
             ylabel='Precio', volume=True, mav=(5, 20),
             show_nontrading=True)

def plot_all_closes_together(df_ohlc):
    df_temp = df_ohlc.reset_index().copy()
    df_close_wide = df_temp.pivot(index='Date', columns='Asset', values='Close')
    df_close_wide.sort_index(inplace=True)

    plt.figure(figsize=(10, 5))
    for asset in df_close_wide.columns:
        plt.plot(df_close_wide.index, df_close_wide[asset], label=asset)
    plt.legend()
    plt.title("Comportamiento conjunto (Close) de todos los activos - Sintético")
    plt.xlabel("Fecha")
    plt.ylabel("Precio Close")
    plt.show()

##############################################################################
# L. MAIN
##############################################################################
if __name__ == "__main__":
    # 1. Descarga tickers S&P500
    sp500_symbols = get_sp500_symbols()
    # Usa unos pocos para ejemplo
    sp500_symbols = sp500_symbols[:5]

    # 2. Descarga de precios
    start_date = "2018-01-01"
    df_sp500 = download_prices(sp500_symbols, start_date=start_date, end_date=None)
    df_sp500 = df_sp500.dropna()

    # 3. Calibra GARCH
    logger.info("Iniciando calibración GARCH + t-Student (con drift y escalado).")
    garch_info = fit_garch_t(df_sp500)
    std_resids = garch_info["residuals"]

    # 4. Ajuste de la T-copula
    logger.info("Iniciando ajuste de T-Copula manual (con mejoras).")
    copula_obj = build_t_copula(
        std_resids,
        nu_global=None,
        use_rank_transform=False,   # puedes poner True si quieres rank transform
        use_higham_pd=False        # puedes poner True para usar Higham
    )

    # 5. Validar la t-copula (Spearman real vs. sintético)
    validate_t_copula(copula_obj, std_resids.values, n_samples=5000)

    # 6. Simular residuos
    n_synthetic_days = 200
    sim_stdres = simulate_correlated_stdresids_tcopula(copula_obj, n_synthetic_days)

    # 7. Reconstruir precios
    last_real_prices = df_sp500.iloc[-1].values
    df_synthetic_close = reconstruct_garch_paths(sim_stdres, garch_info, last_real_prices)

    # Asignamos un rango de fechas sintéticas (solo días hábiles)
    last_date = df_sp500.index[-1]
    dt_range = pd.date_range(last_date + pd.Timedelta(days=1),
                             periods=n_synthetic_days, freq="B")
    df_synthetic_close.index = dt_range

    # 8. Generar OHLC sintético con patrón intradía
    df_ohlc_synthetic = generate_ohlc_from_close(
        df_synthetic_close,
        garch_info["daily_vol"],
        intraday_vol_factor=1.0,
        seed=2024,
        use_intraday_pattern=True
    )

    # 9. Ejemplo de comparación real vs simulado para un activo
    chosen_asset = df_synthetic_close.columns[0]
    real_segment = df_sp500[chosen_asset].iloc[-200:]
    synthetic_segment = df_synthetic_close[chosen_asset]

    plt.figure(figsize=(10, 5))
    plt.plot(real_segment.index, real_segment.values, label=f"{chosen_asset} (Real, últimos 200d)")
    plt.plot(synthetic_segment.index, synthetic_segment.values, label=f"{chosen_asset} (Simulado)")
    plt.title(f"Comparación de Precios para {chosen_asset}")
    plt.legend()
    plt.show()

    # 10. Mostrar la serie simulada OHLC
    asset_ohlc = df_ohlc_synthetic.xs(chosen_asset, level="Asset")
    plt.figure(figsize=(10, 5))
    plt.plot(asset_ohlc.index, asset_ohlc["Close"], label="Close Simulado")
    plt.fill_between(asset_ohlc.index, asset_ohlc["Low"], asset_ohlc["High"],
                     color='gray', alpha=0.3, label="Rango intradía")
    plt.title(f"OHLC Sintético - {chosen_asset}")
    plt.legend()
    plt.show()

    # 11. Candlestick
    plot_ohlc_candlestick(df_ohlc_synthetic, chosen_asset)

    # 12. Todos los closes en una sola gráfica
    plot_all_closes_together(df_ohlc_synthetic)

    logger.info("Fin de la ejecución del script.")
