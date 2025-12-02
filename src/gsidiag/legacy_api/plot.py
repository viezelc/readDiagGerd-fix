#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gsidiag.plotting (legacy-compatible)

This module concentrates most of the **legacy** plotting and quick-analysis
helpers used with GSI diagnostics (both *conventional* and *radiance*).
It preserves the public surface expected by older scripts (e.g., methods
inside :class:`plot_diag`) while adding **NumPy-style docstrings**, inline
comments, and usage examples.

Key ideas
---------
- Works on top of your new unified facade: :func:`readDiag.reader.diagAccess`.
- Accepts both GeoPandas and Cartopy backends for base maps (graceful fallback).
- Provides color helpers (e.g., :func:`getColor`) that return either RGBA bytes
  or hex strings compatible with Matplotlib/GeoPandas.
- Adds robust handling for common pitfalls (missing columns, empty selections,
  absent Cartopy, etc.) without altering legacy signatures.

Notes
-----
- This file intentionally mirrors legacy names like :class:`plot_diag`,
  :meth:`time_series`, :meth:`time_series_radi`, :meth:`statcount` so older
  user scripts continue to run.
- New typed hints and docstrings do **not** change call semantics; they are
  meant to help IDEs, tests, and future maintainers.
- Examples shown in the docstrings assume you already created objects from
  the legacy layer (e.g., ``gd = plot_diag()``) and populated attributes like
  ``obsInfo`` and ``obs`` (which your reader layer prepares).

Examples
--------
Minimal map with points (using GeoPandas fallback if Cartopy is absent):

>>> ax = geoMap(area=[-90, 0, -30, 30])   # South Atlantic box
>>> # df is a GeoDataFrame with 'geometry' (lon/lat) and a value column 'obs'
>>> # df.plot('obs', ax=ax, legend=True)

Full workflow (conventional):
>>> # given a list-like 'cycles' where each item has .obsInfo for one cycle
>>> gd = plot_diag()
>>> # Plot used surface pressure locations (kind 187) colored by 'obs' value
>>> _ = gd.plot('ps', 187, 'obs', mask='iuse==1', area=[-90, 0, -30, 30])

Full workflow (radiance time series by channels):
>>> gd = plot_diag()
>>> # Preserves the legacy outputs (OmF/OmA panels) to PNG files
>>> gd.time_series_radi('amsua', 'n19', dateIni=2024010100, dateFin=2024010300, nHour="06")

"""

from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Core scientific stack
import numpy as np
import pandas as pd
import geopandas as gpd

# New unified facade (already available in your modules)
from readDiag.reader import diagAccess
from readDiag.schema.naming import resolve_col_in_df  # compat resolver
from ..datasources import getVarInfo  # user-provided helper for metadata

# Plotting stack
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.axes import Axes as MplAxes
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker

# Optional Cartopy (fallbacks are handled at runtime)
try:  # pragma: no cover - optional dependency
    from cartopy import crs as ccrs  # noqa: F401
except Exception:  # pragma: no cover
    ccrs = None

# Stdlib
import itertools
import sys
import gc
from datetime import datetime, timedelta
from textwrap import wrap

def _append_colorbar_axes(ax, side: str = "right", size: str = "5%", pad: float = 0.1) -> MplAxes:
    """Cria um eixo de colorbar compatível com GeoAxes (Cartopy)."""
    divider = make_axes_locatable(ax)
    if hasattr(ax, "projection"):  # GeoAxes -> precisa forçar Axes normal
        return divider.append_axes(side, size=size, pad=pad, axes_class=MplAxes)
    return divider.append_axes(side, size=size, pad=pad)

def _get_or_make_cax(ax, side: str = "right", size: str = "5%", pad: float = 0.1) -> MplAxes:
    """Reusa o cax previamente criado para este *ax* (se existir)."""
    cax = getattr(ax, "_gd_cax", None)
    if cax is not None and cax in ax.figure.axes:
        cax.cla()           # limpa para o novo colorbar
        return cax
    cax = _append_colorbar_axes(ax, side=side, size=size, pad=pad)
    ax._gd_cax = cax        # marca para reuso nas próximas chamadas
    return cax

# ---------------------------------------------------------------------------
# Tiny console color helper (legacy formatting kept)
# ---------------------------------------------------------------------------
class setcolor:
    """ANSI color escape sequences used by legacy prints."""

    HEADER    = '\033[95m'
    OKBLUE    = '\033[94m'
    OKGREEN   = '\033[92m'
    WARNING   = '\033[93m'
    FAIL      = '\033[91m'
    ENDC      = '\033[0m'
    BOLD      = '\033[1m'
    UNDERLINE = '\033[4m'


# ---------------------------------------------------------------------------
# Lightweight “help” banner, preserved for legacy parity
# ---------------------------------------------------------------------------
def help() -> None:
    """Print a small Portuguese help banner (legacy stub).

    Examples
    --------
    >>> help()
    Esta é uma ajudada
    """
    print("Esta é uma ajudada")


# ---------------------------------------------------------------------------
# Color utility
# ---------------------------------------------------------------------------
def getColor(
    minVal: float,
    maxVal: float,
    value: float | Iterable[float],
    hex: bool = False,
    cmapName: Optional[str] = None,
):
    """Map a numeric value or sequence to colors within a colormap.

    Parameters
    ----------
    minVal : float
        Minimum data value used to normalize the colormap [vmin].
    maxVal : float
        Maximum data value used to normalize the colormap [vmax].
    value : float or iterable of float
        The value(s) to be mapped onto the colormap domain.
    hex : bool, default False
        If ``True``, return hex strings (e.g., ``"#1f77b4"``).
        If ``False``, return RGBA bytes tuples as expected by some backends.
    cmapName : str, optional
        Matplotlib colormap name. Defaults to ``"Paired"``.

    Returns
    -------
    color : list[str] | list[tuple] | str | tuple
        Either a single color or a list of colors corresponding to ``value``.

    Notes
    -----
    - This function wraps ``matplotlib.cm.get_cmap`` + ``Normalize`` and
      converts the result to either hex strings or RGBA bytes.
    - When ``value`` is scalar, a single color is returned; when iterable,
      the return is a list with one entry per item.

    Examples
    --------
    Map a single number to a hex color:

    >>> getColor(0.0, 10.0, 5.0, hex=True, cmapName="viridis")
    '#...'

    Map a sequence to RGBA (bytes):

    >>> getColor(-1.0, 1.0, [-1, 0, 1], hex=False, cmapName="coolwarm")
    [(..., ..., ..., ...), (...), (...)]
    """
    try:
        import matplotlib.cm as cm
        from matplotlib.colors import Normalize, to_hex
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required for getColor()") from exc

    if cmapName is None:
        cmapName = "Paired"

    # Normalize value(s) into [0, 1]
    norm = Normalize(vmin=float(minVal), vmax=float(maxVal))
    cmap = cm.get_cmap(cmapName)

    def _to_color(x: float):
        rgba = cmap(norm(float(x)))
        return to_hex(rgba) if hex else tuple((np.asarray(rgba) * 255).astype(np.uint8))

    if hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
        return [_to_color(v) for v in value]
    return _to_color(value)  # scalar

# ---------------------------------------------------------------------------
# Base map helper: tries Cartopy features first (robust), then GeoPandas
# ---------------------------------------------------------------------------
def geoMap(area: list[float] | tuple[float, float, float, float] | None = None,
           ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Create or decorate an axes with a geographic background.

    Prefers **Cartopy** features when available (more robust and fast), and
    falls back to **GeoPandas** NaturalEarth outline if Cartopy is missing.
    If neither backend is installed, returns a plain Matplotlib axes.

    Parameters
    ----------
    area : list[float] or tuple[float, float, float, float], optional
        Extent as ``[lon_min, lon_max, lat_min, lat_max]``. If a Cartopy
        GeoAxes is used, ``ax.set_extent(area, PlateCarree())`` is applied;
        otherwise ``xlim/ylim`` are set.
    ax : matplotlib.axes.Axes, optional
        Target axes. If ``None``, tries to create a Cartopy GeoAxes with
        ``ccrs.PlateCarree()``; if it fails, uses ``plt.gca()``.

    Returns
    -------
    matplotlib.axes.Axes
        The axes, decorated with a base map when possible.
    """
    # 1) Create an axes suitable for maps
    if ax is None:
        try:
            import cartopy.crs as _crs
            ax = plt.axes(projection=_crs.PlateCarree())
        except Exception:
            ax = plt.gca()

    # Small helper to test if this axes supports Cartopy features
    def _is_geoaxes(a) -> bool:
        return hasattr(a, "add_feature") and hasattr(a, "set_extent")

    used_backend = None

    # 2) Prefer Cartopy features (fast, no GeoDataFrame instantiation)
    try:
        import cartopy.crs as _crs
        import cartopy.feature as cfeature

        if _is_geoaxes(ax):  # GeoAxes path
            # Land/coastlines/borders with sensible zorder (map under the data)
            ax.add_feature(cfeature.LAND, facecolor="0.95", edgecolor="0.5", linewidth=0.5, zorder=0)
            ax.coastlines(resolution="110m", linewidth=0.6, zorder=1)
            try:
                ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor="0.35", zorder=1)
            except Exception:
                pass

            # Extent handling
            if area is not None:
                ax.set_extent(area, crs=_crs.PlateCarree())
            else:
                # Se nenhuma área foi passada, evita um mapa "vazio"
                ax.set_global()

            # Gridlines com rótulos (em algumas versões campos mudam de nome)
            try:
                gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5, zorder=2)
                for side in ("top", "right"):
                    try:
                        setattr(gl, f"{side}_labels", False)
                    except Exception:
                        pass
            except Exception:
                pass

            used_backend = "cartopy"
        else:
            # axes normal (sem projeção) → tentaremos GeoPandas abaixo
            pass
    except Exception:
        # sem cartopy → tentar GeoPandas
        pass

    # 3) Fallback: GeoPandas NaturalEarth (quando não usamos Cartopy)
    if used_backend is None:
        try:
            import geopandas as _gpd
            # naturalearth_lowres pode faltar em versões novas; tratamos com try/except
            try:
                path = _gpd.datasets.get_path("naturalearth_lowres")
                world = _gpd.read_file(path)
            except Exception:
                world = None

            if world is not None:
                # Traçado do contorno
                try:
                    # Em eixos simples, o plot padrão funciona
                    world.boundary.plot(ax=ax, color="0.3", linewidth=0.6, zorder=0)
                except Exception:
                    world.plot(ax=ax, facecolor="none", edgecolor="0.3", linewidth=0.6, zorder=0)

                # Extent simples via xlim/ylim
                if area is not None:
                    lon_min, lon_max, lat_min, lat_max = area
                    ax.set_xlim(lon_min, lon_max)
                    ax.set_ylim(lat_min, lat_max)

                used_backend = "geopandas"
        except Exception:
            # nenhum backend disponível → segue sem base map
            pass

    # 4) Último recurso: só aplicar extent se nada foi feito acima
    if used_backend is None and area is not None:
        try:
            lon_min, lon_max, lat_min, lat_max = area
            ax.set_xlim(lon_min, lon_max)
            ax.set_ylim(lat_min, lat_max)
        except Exception:
            pass

    return ax


# ---------------------------------------------------------------------------
# Legacy plotting façade (public surface preserved)
# ---------------------------------------------------------------------------
class plot_diag(object):
    """Legacy plotting façade for GSI diagnostics.

    This class bundles quick plotting helpers that act on pre-materialized
    attributes (e.g., ``self.obsInfo``, ``self.obs``). It mirrors the legacy
    public surface so that existing scripts keep working.

    Required attributes (populated externally)
    ------------------------------------------
    obsInfo : dict[str, pd.DataFrame or gpd.GeoDataFrame]
        A mapping from variable name (e.g., ``"ps"``, ``"uv"``, ``"amsua"``) to
        a (Geo)DataFrame. Conventional variables typically expose a MultiIndex
        where level 0 is the *kind* (KX). Radiance may have per-satellite/per-
        channel organizations depending on your upstream loader.
    obs : pd.DataFrame or gpd.GeoDataFrame
        A concatenation of multiple variables for some routines (e.g., :meth:`pvmap`).

    Notes
    -----
    - All methods try to be **robust** to missing columns and to absent optional
      dependencies. Errors are logged in a friendly way and plotting continues
      for what is available.
    - The styling defaults to ``seaborn-v0_8`` where appropriate, to preserve
      the look-and-feel your legacy scripts likely produced.

    Examples
    --------
    Plot used surface pressure values:

    >>> gd = plot_diag()
    >>> # ... set gd.obsInfo = {'ps': gdf_with_points_and_columns}
    >>> ax = gd.plot('ps', 187, 'obs', mask='iuse==1', area=[-90, 0, -30, 30])

    Plot multiple kinds for wind:

    >>> _ = gd.ptmap('uv', [290, 224, 223], marker='.', alpha=0.6)
    """

    # ------------------------------ core scatter/choropleth ------------------------------
    def plot(self, varName, varType, param, minVal=None, maxVal=None, mask=None, area=None, **kwargs):
        """Plot selected observations for a given variable/kind, colored by a column.

        Parameters
        ----------
        varName : str
            Variable key in :attr:`obsInfo` (e.g., ``"ps"``, ``"t"``, ``"uv"``).
        varType : int or str
            Kind (KX) for conventional variables, or first-level selector for radiance.
        param : str
            Column to color the points (e.g., ``"obs"``, ``"omf"``, ``"oma"``).
        minVal, maxVal : float, optional
            Color limits. If both ``None``, the full data range is used.
        mask : str, optional
            Pandas query string (e.g., ``"iuse==1"``). If ``None``, no filtering.
        area : list[float], optional
            Geographic extent as ``[lon_min, lon_max, lat_min, lat_max]``.
        **kwargs
            Forwarded to GeoPandas ``.plot`` (e.g., ``cmap``, ``alpha``, ``markersize``),
            plus:
            - ``style``: Matplotlib style (default ``'seaborn-v0_8'``).
            - ``legend``: bool, create a colorbar inset at the right.

        Returns
        -------
        matplotlib.axes.Axes or None
            The axes used for plotting or ``None`` when data is missing.

        Examples
        --------
        >>> gd = plot_diag()
        >>> ax = gd.plot('ps', 187, 'obs', mask='iuse==1', area=[-90, 0, -30, 30])
        """
        if 'style' in kwargs:
            plt.style.use(kwargs.pop('style'))
        else:
            plt.style.use('seaborn-v0_8')
    
        # ---- 1) Carrega/filtra dados ANTES de criar figura ----
        try:
            df = self.obsInfo[varName].loc[varType]
            if mask is not None:
                df = df.query(mask)
        except Exception as exc:
            print("++++++++++++++++++++++++++ ERROR: file reading --> plot ++++++++++++++++++++++++++")
            print(setcolor.WARNING + f"    >>> No information on this date <<< ({exc})" + setcolor.ENDC)
            return None
    
        if df is None or len(df) == 0:
            print(setcolor.WARNING + "    >>> Empty dataframe for this selection <<<" + setcolor.ENDC)
            return None
    
        # Garante GeoDataFrame com CRS lon/lat
        if not hasattr(df, "geometry"):
            if {"lon", "lat"} <= set(df.columns):
                import geopandas as gpd
                df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326")
            else:
                print(setcolor.WARNING + "    >>> Data must have 'geometry' or 'lon'/'lat' <<<" + setcolor.ENDC)
                return None
    
        # ---- 2) Agora sim cria fig/ax ----
        created_fig = False
        ax = kwargs.pop('ax', None)
        if ax is None:
            try:
                import cartopy.crs as _crs
                fig = plt.figure(figsize=(12, 6))
                ax  = fig.add_subplot(1, 1, 1, projection=_crs.PlateCarree())
            except Exception:
                fig = plt.figure(figsize=(12, 6))
                ax  = fig.add_subplot(1, 1, 1)
            created_fig = True
        else:
            fig = ax.figure
    
        # ---- 3) Infra de colorbar opcional ----
        want_legend = bool(kwargs.pop('legend', False))
        cax = None
        if want_legend:
            cax = _append_colorbar_axes(ax, side="right", size="5%", pad=0.1)
            kwargs['cax'] = cax    

        # ---- 3b) Título: default bonito usando self._idate ----
        from datetime import timezone, datetime as _dt

        def _format_cycle_from_self() -> str:
            """Format cycle from self._idate; assume UTC if naive."""
            dt = getattr(self, "_idate", None)
            if isinstance(dt, _dt):
                if dt.tzinfo is None:
                    # assume UTC when tz-naive
                    return dt.strftime("%Y-%m-%d %H:00 UTC")
                else:
                    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:00 UTC")
            return "cycle unknown"

        # Título do usuário (se passado) ou default elegante
        user_title = kwargs.pop("title", None)

        # Rótulo varType: "KX N" para inteiros; string literal caso contrário
        try:
            import numpy as _np
            is_int_like = isinstance(varType, (int, _np.integer))
        except Exception:
            is_int_like = isinstance(varType, int)
        var_tag = f"{varName} • KX {varType}" if is_int_like else f"{varName} • {varType}"

        cycle_str = _format_cycle_from_self()
        npts = len(df)

        # (Opcional) incluir nome do experimento, se existir:
        # exp = getattr(self, "exp_name", None) or getattr(self, "experiment", None)
        # prefix = f"[{exp}] " if exp else ""

        default_title = f"{var_tag} — {param} — {cycle_str}  (n={npts:,})"
        # default_title = f"{prefix}{var_tag} — {param} — {cycle_str}  (n={npts:,})"  # se usar o prefixo

        ax.set_title(user_title or default_title, loc="left", fontsize=12, fontweight="bold")

    
        kwargs.setdefault('cmap', 'jet')
        kwargs.setdefault('zorder', 10)
        kwargs.setdefault('rasterized', True)  # deixa PDFs/SVGs muito menores
    
        # ---- 4) Basemap robusto ----
        ax = geoMap(area=area, ax=ax)
    
        # transform para GeoAxes/Cartopy
        _transform = None
        try:
            import cartopy.crs as _crs
            if hasattr(ax, "projection"):
                _transform = _crs.PlateCarree()
        except Exception:
            pass
    
        # ---- 5) Plot com fechamento em caso de erro ----
        try:
            if _transform is not None:
                ax = df.plot(column=param, ax=ax, vmin=minVal, vmax=maxVal,
                             transform=_transform,
                             legend=want_legend, legend_kwds={'shrink': 0.5}, **kwargs)
            else:
                ax = df.plot(column=param, ax=ax, vmin=minVal, vmax=maxVal,
                             legend=want_legend, legend_kwds={'shrink': 0.5}, **kwargs)
    
            # layout mais limpo
            try:
                fig.tight_layout()
            except Exception:
                pass
    
            return ax
    
        except Exception as exc:
            # evita deixar a "figura fantasma"
            if created_fig:
                plt.close(fig)
            print("++++++++++++++++++++++++++ ERROR during plotting ++++++++++++++++++++++++++")
            print(setcolor.WARNING + f"    >>> {exc} <<<" + setcolor.ENDC)
            return None


    # ------------------------------ point map across multiple kinds ------------------------------
    def ptmap(self, varName, varType=None, mask=None, area=None, **kwargs):
        """Plot points for a variable optionally across multiple kinds (KX).
    
        Parameters
        ----------
        varName : str
            Variable key (e.g., ``'uv'``, ``'ps'``).
        varType : int or list[int], optional
            One KX or a list of KXs. If ``None``, all kinds for ``varName`` are used.
        mask : str, optional
            Pandas query string (e.g., ``"iuse==1"``).
        area : list[float], optional
            Geographic extent as ``[lon_min, lon_max, lat_min, lat_max]``.
        **kwargs
            Plot options forwarded to GeoPandas ``.plot``; also accepts:
            ``style`` (default ``'seaborn-v0_8'``), ``legend`` (bool).
    
        Returns
        -------
        matplotlib.axes.Axes
            The axes with the plotted points.
    
        Examples
        --------
        >>> gd = plot_diag()
        >>> _ = gd.ptmap('uv', [290, 224, 223], alpha=0.6, marker='.')
        """
        # --- style ---
        plt.style.use(kwargs.pop("style", "seaborn-v0_8"))
    
        # --- axes setup ---
        created_fig = False
        ax = kwargs.pop("ax", None)
        if ax is None:
            try:
                import cartopy.crs as _crs
                fig = plt.figure(figsize=(12, 6))
                ax = fig.add_subplot(1, 1, 1, projection=_crs.PlateCarree())
            except Exception:
                fig = plt.figure(figsize=(12, 6))
                ax = fig.add_subplot(1, 1, 1)
            created_fig = True
        else:
            fig = ax.figure
    
        # --- varType ---
        if varType is None:
            varType = self.obsInfo[varName].index.levels[0].tolist()
        print("varType", varType)
    
        # --- kwargs defaults ---
        kwargs.setdefault("alpha", 0.5)       # transparency
        kwargs.setdefault("marker", "*")      # marker style
        kwargs.setdefault("markersize", 5)    # marker size
        kwargs.setdefault("linewidth", 1)     # line width
    
        # --- legend handling ---
        legend = kwargs.pop("legend", True)   # keep user intent
        kwargs["legend"] = False              # disable auto legend
    
        # --- base map ---
        ax = geoMap(area=area, ax=ax)
    
        # --- cartopy transform (if available) ---
        _transform = None
        try:
            import cartopy.crs as _crs
            if hasattr(ax, "projection"):
                _transform = _crs.PlateCarree()
        except Exception:
            pass
    
        # --- loop over varTypes ---
        cmin, cmax = 0, max(1, len(varType) - 1)
        legend_labels = []
    
        for i, kx in enumerate(varType):
            df = self.obsInfo[varName].loc[kx]
            if mask is not None:
                df = df.query(mask)
    
            color = getColor(minVal=cmin, maxVal=cmax, value=i, hex=True, cmapName="Paired")
            instr = getVarInfo(kx, varName, "instrument")
            label = "\n".join(wrap(f"{varName}-{kx} | {instr}", 30))
    
            legend_labels.append(mpatches.Patch(color=color, label=label))
    
            plot_args = dict(ax=ax, c=color, **kwargs)
            if _transform is not None:
                plot_args["transform"] = _transform
    
            ax = df.plot(**plot_args)
    
        # --- layout ---
        try:
            fig.tight_layout()
        except Exception:
            pass
    
        # --- legend ---
        if legend and legend_labels:
            plt.subplots_adjust(bottom=0.30)
            plt.legend(
                handles=legend_labels,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.08),
                fancybox=False,
                frameon=False,
                numpoints=1,
                prop={"size": 9},
                labelspacing=1.0,
                ncol=4,
            )
    
        return ax

    # ------------------------------ point map across variables using iuse ------------------------------
    def pvmap(self, varName=None, mask=None, area=None, **kwargs):
        """Plot points across variables (e.g., 'uv','ps','t','q') using a common mask.
    
        Parameters
        ----------
        varName : str or list[str], optional
            If ``None``, variables are inferred from :attr:`obs` (most populated first).
            If list, that order is preserved (missing vars are ignored gracefully).
            If str, it is coerced to a single-item list.
        mask : str, optional
            Pandas query string (e.g., ``"iuse==1"``).
        area : list[float], optional
            Geographic extent as ``[lon_min, lon_max, lat_min, lat_max]``.
        **kwargs
            Options forwarded to GeoPandas ``.plot``; accepts:
            - ``style`` (matplotlib style, default ``'seaborn-v0_8'``)
            - ``legend`` (bool, default ``False``)
            - symbol styling such as ``alpha``, ``marker``, ``markersize``, ``linewidth``
    
        Returns
        -------
        matplotlib.axes.Axes
            Axes containing the plot.
        """
        # --- style ---
        plt.style.use(kwargs.pop("style", "seaborn-v0_8"))
    
        # --- axes setup ---
        created_fig = False
        ax = kwargs.pop("ax", None)
        if ax is None:
            try:
                import cartopy.crs as _crs
                fig = plt.figure(figsize=(12, 6))
                ax = fig.add_subplot(1, 1, 1, projection=_crs.PlateCarree())
            except Exception:
                fig = plt.figure(figsize=(12, 6))
                ax = fig.add_subplot(1, 1, 1)
            created_fig = True
        else:
            fig = ax.figure
    
        # --- kwargs defaults ---
        kwargs.setdefault("alpha", 0.5)       # transparency
        kwargs.setdefault("marker", "*")      # marker style
        kwargs.setdefault("markersize", 5)    # marker size
        kwargs.setdefault("linewidth", 1)     # line width
    
        # --- legend handling ---
        legend = kwargs.pop("legend", True)   # keep user intent
        kwargs["legend"] = False              # disable auto legend
    
        # --- base map ---
        ax = geoMap(area=area, ax=ax)
    
        # --- cartopy transform (if available) ---
        _transform = None
        try:
            import cartopy.crs as _crs
            if hasattr(ax, "projection"):
                _transform = _crs.PlateCarree()
        except Exception:
            pass
    
        # --- infer variable order ---
        # self.obs is assumed to have a MultiIndex with level 0 = variable key
        total = self.obs.groupby(level=0).size()
    
        if varName is None:
            # Most populated variables first
            vars_ = list(total.sort_values(ascending=False).index)
        elif isinstance(varName, str):
            vars_ = [varName]
        else:
            # Preserve user order but drop missing keys gracefully
            vars_ = [v for v in varName if v in total.index] or list(varName)
    
    
        # --- color palette (stable cycling) ---
        colors_palette = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
            "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"
        ]
    
        legend_handles = []
    
        # --- plot each variable with a distinct color ---
        for i, v in enumerate(vars_):
            # Guard against missing keys in obsInfo
            if v not in self.obsInfo:
                continue
    
            df = self.obsInfo[v]
            if mask is not None:
                df = df.query(mask)
    
            color = colors_palette[i % len(colors_palette)]
            legend_handles.append(mpatches.Patch(color=color, label=v))
    
            plot_args = dict(ax=ax, c=color, **kwargs)
            if _transform is not None:
                plot_args["transform"] = _transform
    
            ax = df.plot(**plot_args)
    
        # --- layout once (after loop) ---
        try:
            fig.tight_layout()
        except Exception:
            pass
    
        # --- optional legend ---
        if legend and legend_handles:
            plt.legend(
                handles=legend_handles,
                numpoints=1,
                loc="best",
                bbox_to_anchor=(1.1, 0.6),
                frameon=False,
                ncol=1,
                prop={"size": 10},
            )
    
        return ax

    # ------------------------------ counts by KX for a variable ------------------------------
    def pcount(self, varName, **kwargs):
        """Bar chart of counts per kind (KX) for a given variable.

        Parameters
        ----------
        varName : str
            Variable key in :attr:`obsInfo`.
        **kwargs
            Forwarded to Pandas/Matplotlib ``.plot.bar``. Also accepts ``style``.

        Examples
        --------
        >>> gd = plot_diag()
        >>> gd.pcount('ps', rot=45)
        """
        try:
            import matplotlib.cm as cm  # noqa: F401
            from matplotlib.colors import Normalize  # noqa: F401
        except ImportError:  # pragma: no cover
            pass

        if 'style' in kwargs:
            plt.style.use(kwargs.pop('style'))
        else:
            plt.style.use('seaborn-v0_8')

        kwargs.setdefault('alpha', 0.5)
        kwargs.setdefault('rot', 45)
        kwargs.setdefault('legend', False)

        df = self.obsInfo[varName].groupby(level=0).size()

        colors = getColor(minVal=float(df.min()), maxVal=float(df.max()),
                          value=df.values, hex=True, cmapName='Paired')

        df.plot.bar(color=colors, **kwargs)
        plt.ylabel('Number of Observations')
        plt.xlabel('KX')
        plt.title('Variable Name : ' + varName)

    # ------------------------------ impacts (conventional/radiance) ------------------------------
    def impConv(self, varName):
        """Horizontal bar plot of average impacts by KX (conventional).

        Parameters
        ----------
        varName : str
            Conventional variable key (e.g., ``'t'``, ``'q'``).

        Examples
        --------
        >>> gd = plot_diag()
        >>> gd.impConv('t')
        """
        try:
            import seaborn as sb
        except ImportError:  # pragma: no cover
            raise RuntimeError("seaborn is required for impConv()")

        df0 = self.obsInfo[varName].drop(columns='geometry', errors='ignore')
        df = df0.groupby(level=0).mean(numeric_only=True).reset_index()

        sb.barplot(data=df, x='imp', y='kx', orient='h', errorbar=None, color="darkseagreen")
        plt.ylabel('Mnemonics')
        plt.xlabel('Impact of observations')
        plt.title('Variable Name : ' + varName)

    def impRad(self, varName):
        """Horizontal bar plot of average impacts by satellite id (radiance).

        Parameters
        ----------
        varName : str
            Radiance variable key (e.g., ``'amsua'``).

        Examples
        --------
        >>> gd = plot_diag()
        >>> gd.impRad('amsua')
        """
        try:
            import seaborn as sb
        except ImportError:  # pragma: no cover
            raise RuntimeError("seaborn is required for impRad()")

        df0 = self.obsInfo[varName].drop(columns='geometry', errors='ignore')
        df = df0.groupby(level=0).mean(numeric_only=True).reset_index()

        sb.barplot(data=df, x='imp', y='SatId', orient='h', errorbar=None, color="darkcyan")
        plt.ylabel('Satellite_ID')
        plt.xlabel('Impact of observations')
        plt.title('Variable Name : ' + varName)

    def ibfConv(self, varName):
        """Fraction of beneficial impacts (impact < 0) by KX (conventional).

        Parameters
        ----------
        varName : str
            Conventional variable key.

        Notes
        -----
        Fraction is computed relative to the total count per KX.

        Examples
        --------
        >>> gd = plot_diag()
        >>> gd.ibfConv('q')
        """
        try:
            import seaborn as sb
        except ImportError:  # pragma: no cover
            raise RuntimeError("seaborn is required for ibfConv()")

        df0 = self.obsInfo[varName].drop(columns='geometry', errors='ignore')
        df = ((df0.query('imp<0').groupby('kx').size() / df0.groupby('kx').size()) * 100).reset_index(name='ibf')

        sb.barplot(data=df, x='ibf', y='kx', orient='h', errorbar=None, color="aqua")
        plt.ylabel('Mnemonics')
        plt.xlabel('Fractional beneficial impact')
        plt.title('Variable Name : ' + varName)

    def ibfRad(self, varName):
        """Fraction of beneficial impacts (impact < 0) by satellite id (radiance).

        Parameters
        ----------
        varName : str
            Radiance variable key.

        Examples
        --------
        >>> gd = plot_diag()
        >>> gd.ibfRad('amsua')
        """
        try:
            import seaborn as sb
        except ImportError:  # pragma: no cover
            raise RuntimeError("seaborn is required for ibfRad()")

        df0 = self.obsInfo[varName].drop(columns='geometry', errors='ignore')
        df = ((df0.query('imp<0').groupby('SatId').size() / df0.groupby('SatId').size()) * 100).reset_index(name='ibf')

        sb.barplot(data=df, x='ibf', y='SatId', orient='h', errorbar=None, color="aqua")
        plt.ylabel('Satellite_ID')
        plt.xlabel('Fractional beneficial impact')
        plt.title('Variable Name : ' + varName)

    # ------------------------------ global counts across variables and by KX ------------------------------
    def vcount(self, **kwargs):
        """Bar chart of **total** observations by variable (across kinds).

        Parameters
        ----------
        **kwargs
            Forwarded to Pandas/Matplotlib ``.plot.bar``. Accepts ``style``.

        Examples
        --------
        >>> gd = plot_diag()
        >>> gd.vcount(rot=0)
        """
        if 'style' in kwargs:
            plt.style.use(kwargs.pop('style'))
        else:
            plt.style.use('seaborn-v0_8')

        kwargs.setdefault('alpha', 0.5)
        kwargs.setdefault('rot', 0)
        kwargs.setdefault('legend', False)

        df = pd.DataFrame({key: len(value) for key, value in self.obsInfo.items()}, index=['total']).T

        colors = getColor(minVal=float(df['total'].min()), maxVal=float(df['total'].max()),
                          value=df['total'].values, hex=True, cmapName='Paired')

        df.plot.bar(color=colors, **kwargs)
        plt.ylabel('Number of Observations')
        plt.xlabel('Variable Names')
        plt.title('Total Number of Observations')

    def kxcount(self, **kwargs):
        """Bar chart of **total** observations grouped by KX.

        Parameters
        ----------
        **kwargs
            Forwarded to Pandas/Matplotlib ``.plot.bar``. Accepts ``style``.

        Examples
        --------
        >>> gd = plot_diag()
        >>> gd.kxcount(rot=90)
        """
        if 'style' in kwargs:
            plt.style.use(kwargs.pop('style'))
        else:
            plt.style.use('seaborn-v0_8')

        kwargs.setdefault('alpha', 0.5)
        kwargs.setdefault('rot', 90)
        kwargs.setdefault('legend', False)

        df = self.obs.groupby(self.obs.index.get_level_values("kx")).size()

        colors = getColor(minVal=float(df.min()), maxVal=float(df.max()),
                          value=df.values, hex=True, cmapName='Paired')

        df.plot.bar(color=colors, **kwargs)
        plt.ylabel('Number of Observations by KX')
        plt.xlabel('KX number')
        plt.title('Total Number of Observations')

               
 
    def time_series(self, varName=None, varType=None, mask=None, dateIni=None, dateFin=None, nHour="06", vminOMA=None, vmaxOMA=None, vminSTD=0.0, vmaxSTD=14.0, Level=None, Lay = None, SingleL=None, Clean=None):
        
        '''
        The time_series function plots a time series for different levels/layers or for a single level/layer considering
        OmF and OmA. 

        Example:

        vName = 'uv'          # Variable
        vType = 224           # Source Type
        mask  = None          # Mask the data by chosen used/not used data, ex: mask='iuse==1'
        dateIni = 2013010100  # Inicial Date
        dateFin = 2013010900  # Final Date
        nHour = "06"          # Time Interval
        vminOMA = -4.0        # Y-axis Minimum Value for OmF or OmA
        vmaxOMA = 4.0         # Y-axis Maximum Value for OmF or OmA
        vminSTD = 0.0         # Y-axis Minimum Value for Standard Deviation
        vmaxSTD = 14.0        # Y-axis Maximum Value for Standard Deviation
        Level = 1000          # Time Series Level, if any (None), all standard levels are plotted
        Lay = 15              # The size of half layer in hPa, if the plot type is sampled by layers.
        SingleL = "OneL"      # When level is fixed, ex: 1000 hPa, the plot can be exactly in this level (SingleL = None),
                              # on all levels as a single layer (SingleL = "All") or on a layer centered in Level and bounded by
                              # Level-Lay and Level+Lay (SingleL="OneL"). If Lay is not defined, it will be used a standard value of 50 hPa. 

        '''
        if Clean == None:
            Clean = True

        delta = nHour
        omflag = "OmF"
        omflaga = "OmA"

        Laydef = 50

        separator = " ====================================================================================================="

        print()
        print(separator)
        #print(" Reading dataset in " + data_path)
        varInfo = getVarInfo(varType, varName, 'instrument')
        if varInfo is not None:
            print(" Analyzing data of variable: " + varName + "  ||  type: " + str(varType) + "  ||  " + varInfo + "  ||  check: " + omflag)
        else:
            print(" Analyzing data of variable: " + varName + "  ||  type: " + str(varType) + "  ||  Unknown instrument  ||  check: " + omflag)

        print(separator)
        print()

        if mask == None:
            maski  = "iuse>-99999.9"
            cmaski = "iuse = All"
        else:
            maski  = mask
            cmaski = mask

        if type(Level) == list:
            zlevs_def = Level
            Level = "Zlevs"
        else:
            zlevs_def = list(map(int,self[0].zlevs))

        print(zlevs_def)

        datei = datetime.strptime(str(dateIni), "%Y%m%d%H")
        datef = datetime.strptime(str(dateFin), "%Y%m%d%H")
        date  = datei

        levs_tmp, DayHour_tmp = [], []
        info_check = {}
        f = 0
        while (date <= datef):
            
            datefmt = date.strftime("%Y%m%d%H")
            DayHour_tmp.append(date.strftime("%d%H"))
            
            dataDict = self[f].obsInfo[varName].query(maski).loc[varType]
            info_check.update({date.strftime("%d%H"):True})

            if 'prs' in dataDict and (Level == None or Level == "Zlevs"):
                if(Level == None):
                    levs_tmp.extend(list(set(map(int,dataDict['prs']))))
                else:
                    levs_tmp = zlevs_def[::-1]
                info_check.update({date.strftime("%d%H"):True})
                print(date.strftime(' Preparing data for: ' + "%Y-%m-%d:%H"))
                print(' Levels: ', sorted(levs_tmp), end='\n')
                print("")
                f = f + 1
            else:
                if (Level != None and Level != "Zlevs") and info_check[date.strftime("%d%H")] == True:
                    levs_tmp.extend([Level])
                    print(date.strftime(' Preparing data for: ' + "%Y-%m-%d:%H"), ' - Level: ', Level , end='\n')
                    f = f + 1
                else:
                    info_check.update({date.strftime("%d%H"):False})
                    print(date.strftime(setcolor.WARNING + ' Preparing data for: ' + "%Y-%m-%d:%H"), ' - No information on this date ' + setcolor.ENDC, end='\n')

            del(dataDict)
            
            date = date + timedelta(hours=int(delta))
            
        if(len(DayHour_tmp) > 4):
            DayHour = [hr if (ix % int(len(DayHour_tmp) / 4)) == 0 else '' for ix, hr in enumerate(DayHour_tmp)]
        else:
            DayHour = DayHour_tmp

        zlevs = [z if z in zlevs_def else "" for z in sorted(set(levs_tmp+zlevs_def))]

        print()
        print(separator)
        print()

        list_meanByLevs, list_stdByLevs, list_countByLevs = [], [], []
        list_meanByLevsa, list_stdByLevsa, list_countByLevsa = [], [], []
        date = datei
        levs = sorted(list(set(levs_tmp)))
        levs_tmp.clear()
        del(levs_tmp[:])

        f = 0
        while (date <= datef):

            print(date.strftime(' Calculating for ' + "%Y-%m-%d:%H"))
            datefmt = date.strftime("%Y%m%d%H")

            try: 
                if info_check[date.strftime("%d%H")] == True:
                    dataDict = self[f].obsInfo[varName].query(maski).loc[varType]
                    dataByLevs, mean_dataByLevs, std_dataByLevs, count_dataByLevs = {}, {}, {}, {}
                    dataByLevsa, mean_dataByLevsa, std_dataByLevsa, count_dataByLevsa = {}, {}, {}, {}
                    [dataByLevs.update({int(lvl): []}) for lvl in levs]
                    [dataByLevsa.update({int(lvl): []}) for lvl in levs]
                    if Level != None and Level != "Zlevs":
                        if SingleL == None:
                            [ dataByLevs[int(p)].append(v) for p,v in zip(self[f].obsInfo[varName].query(maski).loc[varType].prs,self[f].obsInfo[varName].query(maski).loc[varType].omf) if int(p) == Level ]
                            [ dataByLevsa[int(p)].append(v) for p,v in zip(self[f].obsInfo[varName].query(maski).loc[varType].prs,self[f].obsInfo[varName].query(maski).loc[varType].oma) if int(p) == Level ]
                            forplot = ' Level='+str(Level) +'hPa'
                            forplotname = 'level_'+str(Level) +'hPa'
                        else:
                            if SingleL == "All":
                                [ dataByLevs[Level].append(v) for v in self[f].obsInfo[varName].query(maski).loc[varType].omf ]
                                [ dataByLevsa[Level].append(v) for v in self[f].obsInfo[varName].query(maski).loc[varType].oma ]
                                forplot = ' Layer=Entire Atmosphere'
                                forplotname = 'layer_allAtm'
                            else:
                                if SingleL == "OneL":
                                    if Lay == None:
                                        print("")
                                        print(" Variable Lay is None, resetting it to its default value: "+str(Laydef)+" hPa.")
                                        print("")
                                        Lay = Laydef
                                    [ dataByLevs[int(Level)].append(v) for p,v in zip(self[f].obsInfo[varName].query(maski).loc[varType].prs,self[f].obsInfo[varName].query(maski).loc[varType].omf) if int(p) >=Level-Lay and int(p) <Level+Lay ]
                                    [ dataByLevsa[int(Level)].append(v) for p,v in zip(self[f].obsInfo[varName].query(maski).loc[varType].prs,self[f].obsInfo[varName].query(maski).loc[varType].oma) if int(p) >=Level-Lay and int(p) <Level+Lay ]
                                    forplot = ' Layer='+str(Level+Lay)+'-'+str(Level-Lay)+'hPa'
                                    forplotname = 'layer_'+str(Level+Lay)+'-'+str(Level-Lay)+'hPa'
                                else:
                                    print(" Wrong value for variable SingleL. Please, check it and rerun the script.")    
                    else:
                        if Level == None:
                            [ dataByLevs[int(p)].append(v) for p,v in zip(self[f].obsInfo[varName].query(maski).loc[varType].prs,self[f].obsInfo[varName].query(maski).loc[varType].omf) ]
                            [ dataByLevsa[int(p)].append(v) for p,v in zip(self[f].obsInfo[varName].query(maski).loc[varType].prs,self[f].obsInfo[varName].query(maski).loc[varType].oma) ]
                            forplotname = 'all_levels_byLevels'
                        else:
                            for ll in range(len(levs)):
                                lv = levs[ll]
                                if Lay == None:
                                    if ll == 0:
                                        Llayi = 0
                                    else:
                                        Llayi = (levs[ll] - levs[ll-1]) / 2.0
                                    if ll == len(levs)-1:
                                        Llayf = Llayi
                                    else:
                                        Llayf = (levs[ll+1] - levs[ll]) / 2.0
                                    cutlevs = [ v for p,v in zip(self[f].obsInfo[varName].query(maski).loc[varType].prs,self[f].obsInfo[varName].query(maski).loc[varType].omf) if int(p) >=lv-Llayi and int(p) <lv+Llayf ]
                                    cutlevsa = [ v for p,v in zip(self[f].obsInfo[varName].query(maski).loc[varType].prs,self[f].obsInfo[varName].query(maski).loc[varType].oma) if int(p) >=lv-Llayi and int(p) <lv+Llayf ]
                                    forplotname = 'all_levels_filledLayers'
                                else:
                                    cutlevs = [ v for p,v in zip(self[f].obsInfo[varName].query(maski).loc[varType].prs,self[f].obsInfo[varName].query(maski).loc[varType].omf) if int(p) >=lv-Lay and int(p) <lv+Lay ]
                                    cutlevsa = [ v for p,v in zip(self[f].obsInfo[varName].query(maski).loc[varType].prs,self[f].obsInfo[varName].query(maski).loc[varType].oma) if int(p) >=lv-Lay and int(p) <lv+Lay ]
                                    forplotname = 'all_levels_bylayers_'+str(Lay)+"hPa"
                                [ dataByLevs[lv].append(il) for il in cutlevs ]
                                [ dataByLevsa[lv].append(il) for il in cutlevsa ]
                    f = f + 1
                for lv in levs:
                    if len(dataByLevs[lv]) != 0 and info_check[date.strftime("%d%H")] == True:
                        mean_dataByLevs.update({int(lv): np.mean(np.array(dataByLevs[lv]))})
                        std_dataByLevs.update({int(lv): np.std(np.array(dataByLevs[lv]))})
                        count_dataByLevs.update({int(lv): len(np.array(dataByLevs[lv]))})
                        mean_dataByLevsa.update({int(lv): np.mean(np.array(dataByLevsa[lv]))})
                        std_dataByLevsa.update({int(lv): np.std(np.array(dataByLevsa[lv]))})
                        count_dataByLevsa.update({int(lv): len(np.array(dataByLevsa[lv]))})
                    else:
                        mean_dataByLevs.update({int(lv): -99})
                        std_dataByLevs.update({int(lv): -99})
                        count_dataByLevs.update({int(lv): -99})
                        mean_dataByLevsa.update({int(lv): -99})
                        std_dataByLevsa.update({int(lv): -99})
                        count_dataByLevsa.update({int(lv): -99})
            
            except:
                if info_check[date.strftime("%d%H")] == True:
                    print("ERROR in time_series function.")
                else:
                    print(setcolor.WARNING + "    >>> No information on this date (" + str(date.strftime("%Y-%m-%d:%H")) +") <<< " + setcolor.ENDC)

                for lv in levs:
                    mean_dataByLevs.update({int(lv): -99})
                    std_dataByLevs.update({int(lv): -99})
                    count_dataByLevs.update({int(lv): -99})
                    mean_dataByLevsa.update({int(lv): -99})
                    std_dataByLevsa.update({int(lv): -99})
                    count_dataByLevsa.update({int(lv): -99})

            if Level == None or Level == "Zlevs":
                list_meanByLevs.append(list(mean_dataByLevs.values()))
                list_stdByLevs.append(list(std_dataByLevs.values()))
                list_countByLevs.append(list(count_dataByLevs.values()))
                list_meanByLevsa.append(list(mean_dataByLevsa.values()))
                list_stdByLevsa.append(list(std_dataByLevsa.values()))
                list_countByLevsa.append(list(count_dataByLevsa.values()))
            else:
                list_meanByLevs.append(mean_dataByLevs[int(Level)])
                list_stdByLevs.append(std_dataByLevs[int(Level)])
                list_countByLevs.append(count_dataByLevs[int(Level)])
                list_meanByLevsa.append(mean_dataByLevsa[int(Level)])
                list_stdByLevsa.append(std_dataByLevsa[int(Level)])
                list_countByLevsa.append(count_dataByLevsa[int(Level)])

            dataByLevs.clear()
            mean_dataByLevs.clear()
            std_dataByLevs.clear()
            count_dataByLevs.clear()
            dataByLevsa.clear()
            mean_dataByLevsa.clear()
            std_dataByLevsa.clear()
            count_dataByLevsa.clear()

            date_finale = date
            date = date + timedelta(hours=int(delta))

        print()
        print(separator)
        print()

        print(' Making Graphics...')

        y_axis      = np.arange(0, len(zlevs), 1)
        x_axis      = np.arange(0, len(DayHour), 1)

        mean_final  = np.ma.masked_array(np.array(list_meanByLevs), np.array(list_meanByLevs) == -99)
        std_final   = np.ma.masked_array(np.array(list_stdByLevs), np.array(list_stdByLevs) == -99)
        count_final = np.ma.masked_array(np.array(list_countByLevs), np.array(list_countByLevs) == -99)
        mean_finala  = np.ma.masked_array(np.array(list_meanByLevsa), np.array(list_meanByLevsa) == -99)
        std_finala   = np.ma.masked_array(np.array(list_stdByLevsa), np.array(list_stdByLevsa) == -99)
        count_finala = np.ma.masked_array(np.array(list_countByLevsa), np.array(list_countByLevsa) == -99)

        OMF_inf = np.array(list_meanByLevs)-np.array(list_stdByLevs)
        OMF_sup = np.array(list_meanByLevs)+np.array(list_stdByLevs)
        OMA_inf = np.array(list_meanByLevsa)-np.array(list_stdByLevsa)
        OMA_sup = np.array(list_meanByLevsa)+np.array(list_stdByLevsa)

        mean_limit_inf = np.min(np.array([np.min(mean_final), np.min(mean_finala)]))
        mean_limit_sup = np.max(np.array([np.max(mean_final), np.max(mean_finala)]))

        std_limit_inf = np.min(np.array([np.min(std_final), np.min(std_finala)]))
        std_limit_sup = np.max(np.array([np.max(std_final), np.max(std_finala)]))

        omfoma_limit_inf =     (np.min(np.array([np.min(OMF_inf), np.min(OMA_inf)])))
        if omfoma_limit_inf > 0:
            omfoma_limit_inf = 0.9*omfoma_limit_inf
        else:
            omfoma_limit_inf = 1.1*omfoma_limit_inf  
        omfoma_limit_sup = 1.1*(np.max(np.array([np.max(OMF_sup), np.max(OMA_sup)])))

        if (vminOMA == None) and (vmaxOMA == None): vminOMA, vmaxOMA = mean_limit_inf, 1.1*mean_limit_sup
        if vminOMA > 0:
            vminOMA = 0.9*vminOMA
        else:
            vminOMA = 1.1*vminOMA 

        vmaxOMAabs = np.max([np.abs(vminOMA),np.abs(vminOMA)])

        if (vminSTD == None) and (vmaxSTD == None): vminSTD, vmaxSTD = std_limit_inf - 0.1*std_limit_inf,  1.1*std_limit_sup

        date_title = str(datei.strftime("%d%b")) + '-' + str(date_finale.strftime("%d%b")) + ' ' + str(date_finale.strftime("%Y"))
        instrument_title = str(varName) + '-' + str(varType) + '  |  ' + getVarInfo(varType, varName, 'instrument')

        # Figure with more than one level - default levels: [600, 700, 800, 900, 1000]
        if Level == None or Level == "Zlevs":
            fig = plt.figure(figsize=(6, 9))
            plt.rcParams['axes.facecolor'] = 'None'
            plt.rcParams['hatch.linewidth'] = 0.3

            ##### OMF

            plt.subplot(3, 1, 1)
            ax = plt.gca()
            ax.add_patch(mpl.patches.Rectangle((-1,-1),(len(DayHour)+1),(len(levs)+3), hatch='xxxxx', color='black', fill=False, snap=False, zorder=0))
            plt.imshow(np.flipud(mean_final.T), origin='lower', vmin=-vmaxOMAabs, vmax=vmaxOMAabs, cmap='seismic', aspect='auto', zorder=1,interpolation='none')
            plt.colorbar(orientation='horizontal', pad=0.18, shrink=1.0)
            plt.tight_layout()
            plt.title(instrument_title, loc='left', fontsize=10)
            plt.title(date_title, loc='right', fontsize=10)
            plt.ylabel('Vertical Levels (hPa)')
            plt.xlabel('Mean ('+omflag+')', labelpad=50)
            plt.yticks(y_axis, zlevs[::-1])
            plt.xticks(x_axis, DayHour)
            major_ticks = [ DayHour.index(dh) for dh in filter(None,DayHour) ]
            ax.set_xticks(major_ticks)

            plt.subplot(3, 1, 2)
            ax = plt.gca()
            ax.add_patch(mpl.patches.Rectangle((-1,-1),(len(DayHour)+1),(len(levs)+3), hatch='xxxxx', color='black', fill=False, snap=False, zorder=0))
            plt.imshow(np.flipud(std_final.T), origin='lower', vmin=vminSTD, vmax=vmaxSTD, cmap='Blues', aspect='auto', zorder=1,interpolation='none')
            plt.colorbar(orientation='horizontal', pad=0.18, shrink=1.0)
            plt.tight_layout()
            plt.title(instrument_title, loc='left', fontsize=10)
            plt.title(date_title, loc='right', fontsize=10)
            plt.ylabel('Vertical Levels (hPa)')
            plt.xlabel('Standard Deviation ('+omflag+')', labelpad=50)
            plt.yticks(y_axis, zlevs[::-1])
            plt.xticks(x_axis, DayHour)
            major_ticks = [ DayHour.index(dh) for dh in filter(None,DayHour) ]
            ax.set_xticks(major_ticks)

            plt.subplot(3, 1, 3)
            ax = plt.gca()
            ax.add_patch(mpl.patches.Rectangle((-1,-1),(len(DayHour)+1),(len(levs)+3), hatch='xxxxx', color='black', fill=False, snap=False, zorder=0))
            plt.imshow(np.flipud(count_final.T), origin='lower', vmin=0.0, vmax=np.max(count_final), cmap='gist_heat_r', aspect='auto', zorder=1,interpolation='none')
            plt.colorbar(orientation='horizontal', pad=0.18, shrink=1.0)
            plt.title(instrument_title, loc='left', fontsize=10)
            plt.title(date_title, loc='right', fontsize=10)
            plt.ylabel('Vertical Levels (hPa)')
            plt.xlabel('Total Observations'+" ("+cmaski+")", labelpad=50)
            plt.yticks(y_axis, zlevs[::-1])
            plt.xticks(x_axis, DayHour)
            major_ticks = [ DayHour.index(dh) for dh in filter(None,DayHour) ]
            ax.set_xticks(major_ticks)

            plt.tight_layout()
            plt.savefig('time_series_'+str(varName) + '-' + str(varType)+'_'+omflag+'_'+forplotname+'.png', bbox_inches='tight', dpi=100)
            if Clean:
                plt.clf()

            ##### OMA

            fig = plt.figure(figsize=(6, 9))
            plt.rcParams['axes.facecolor'] = 'None'
            plt.rcParams['hatch.linewidth'] = 0.3

            plt.subplot(3, 1, 1)
            ax = plt.gca()
            ax.add_patch(mpl.patches.Rectangle((-1,-1),(len(DayHour)+1),(len(levs)+3), hatch='xxxxx', color='black', fill=False, snap=False, zorder=0))
            plt.imshow(np.flipud(mean_finala.T), origin='lower', vmin=-vmaxOMAabs, vmax=vmaxOMAabs, cmap='seismic', aspect='auto', zorder=1,interpolation='none')
            plt.colorbar(orientation='horizontal', pad=0.18, shrink=1.0)
            plt.tight_layout()
            plt.title(instrument_title, loc='left', fontsize=10)
            plt.title(date_title, loc='right', fontsize=10)
            plt.ylabel('Vertical Levels (hPa)')
            plt.xlabel('Mean ('+omflaga+')', labelpad=50)
            plt.yticks(y_axis, zlevs[::-1])
            plt.xticks(x_axis, DayHour)
            major_ticks = [ DayHour.index(dh) for dh in filter(None,DayHour) ]
            ax.set_xticks(major_ticks)

            plt.subplot(3, 1, 2)
            ax = plt.gca()
            ax.add_patch(mpl.patches.Rectangle((-1,-1),(len(DayHour)+1),(len(levs)+3), hatch='xxxxx', color='black', fill=False, snap=False, zorder=0))
            plt.imshow(np.flipud(std_finala.T), origin='lower', vmin=vminSTD, vmax=vmaxSTD, cmap='Blues', aspect='auto', zorder=1,interpolation='none')
            plt.colorbar(orientation='horizontal', pad=0.18, shrink=1.0)
            plt.tight_layout()
            plt.title(instrument_title, loc='left', fontsize=10)
            plt.title(date_title, loc='right', fontsize=10)
            plt.ylabel('Vertical Levels (hPa)')
            plt.xlabel('Standard Deviation ('+omflaga+')', labelpad=50)
            plt.yticks(y_axis, zlevs[::-1])
            plt.xticks(x_axis, DayHour)
            major_ticks = [ DayHour.index(dh) for dh in filter(None,DayHour) ]
            ax.set_xticks(major_ticks)

            plt.subplot(3, 1, 3)
            ax = plt.gca()
            ax.add_patch(mpl.patches.Rectangle((-1,-1),(len(DayHour)+1),(len(levs)+3), hatch='xxxxx', color='black', fill=False, snap=False, zorder=0))
            plt.imshow(np.flipud(count_finala.T), origin='lower', vmin=0.0, vmax=np.max(count_finala), cmap='gist_heat_r', aspect='auto', zorder=1,interpolation='none')
            plt.colorbar(orientation='horizontal', pad=0.18, shrink=1.0)
            plt.title(instrument_title, loc='left', fontsize=10)
            plt.title(date_title, loc='right', fontsize=10)
            plt.ylabel('Vertical Levels (hPa)')
            plt.xlabel('Total Observations'+" ("+cmaski+")", labelpad=50)
            plt.yticks(y_axis, zlevs[::-1])
            plt.xticks(x_axis, DayHour)
            major_ticks = [ DayHour.index(dh) for dh in filter(None,DayHour) ]
            ax.set_xticks(major_ticks)

            plt.tight_layout()
            plt.savefig('time_series_'+str(varName) + '-' + str(varType)+'_'+omflaga+'_'+forplotname+'.png', bbox_inches='tight', dpi=100)
            if Clean:
                plt.clf()

        # Figure with only one level
        else:
        
            ##### OMF

            fig = plt.figure(figsize=(6, 4))
            fig, ax1 = plt.subplots(1, 1)
            plt.style.use('seaborn-v0_8-ticks')

            plt.axhline(y=0.0,ls='solid',c='#d3d3d3')
            plt.annotate(forplot, xy=(0.0, 0.965), xytext=(0,0), xycoords='axes fraction', textcoords='offset points', color='lightgray', fontweight='bold', fontsize='12',
            horizontalalignment='left', verticalalignment='center')

            ax1.plot(x_axis, list_meanByLevs, "b-", label="Mean ("+omflag+")")
            ax1.plot(x_axis, list_meanByLevs, "bo", label="Mean ("+omflag+")")
            ax1.set_xlabel('Date (DayHour)', fontsize=10)
            # Make the y-axis label, ticks and tick labels match the line color.
            ax1.set_ylim(vminOMA, vmaxOMA)
            ax1.set_ylabel('Mean ('+omflag+')', color='b', fontsize=10)
            ax1.tick_params('y', colors='b')
            plt.xticks(x_axis, DayHour)
            major_ticks = [ DayHour.index(dh) for dh in filter(None,DayHour) ]
            ax1.set_xticks(major_ticks)
            plt.axhline(y=np.mean(list_meanByLevs),ls='dotted',c='blue')
            
            ax2 = ax1.twinx()
            ax2.plot(x_axis, std_final, "r-", label="Std. Deviation ("+omflag+")")
            ax2.plot(x_axis, std_final, "rs", label="Std. Deviation ("+omflag+")")
            ax2.set_ylim(vminSTD, vmaxSTD)
            ax2.set_ylabel('Std. Deviation ('+omflag+')', color='r', fontsize=10)
            ax2.tick_params('y', colors='r')
            major_ticks = np.arange(0, max(x_axis), len(DayHour)/len(list(filter(None, DayHour))))
            ax2.set_xticks(major_ticks)
            plt.axhline(y=np.mean(std_final),ls='dotted',c='red')

            ax3 = ax1.twinx()
            ax3.plot(x_axis, count_final, "g-", label="Total Observations"+" ("+cmaski+")")
            ax3.plot(x_axis, count_final, "g^", label="Total Observations"+" ("+cmaski+")")
            ax3.set_ylim(0, np.max(count_final) + (np.max(count_final)/8))
            ax3.set_ylabel('Total Observations'+" ("+cmaski+")", color='g', fontsize=10)
            ax3.tick_params('y', colors='g')
            ax3.spines["right"].set_position(("axes", 1.15))
            plt.yticks(rotation=90)
            plt.axhline(y=np.mean(count_final),ls='dotted',c='green')

            ax3.set_title(instrument_title, loc='left', fontsize=10)
            ax3.set_title(date_title, loc='right', fontsize=10)

            plt.xticks(x_axis, DayHour)
            major_ticks = [ DayHour.index(dh) for dh in filter(None,DayHour) ]
            ax3.set_xticks(major_ticks)
            plt.title(instrument_title, loc='left', fontsize=9)
            plt.title(date_title, loc='right', fontsize=9)
            plt.subplots_adjust(left=None, bottom=None, right=0.80, top=None)
            plt.tight_layout()
            plt.savefig('time_series_'+str(varName) + '-' + str(varType)+'_'+omflag+'_'+forplotname+'.png', bbox_inches='tight', dpi=100)
            if Clean:
                plt.clf()

            ##### OMA

            fig = plt.figure(figsize=(6, 4))
            fig, ax1 = plt.subplots(1, 1)
            plt.style.use('seaborn-v0_8-ticks')

            plt.axhline(y=0.0,ls='solid',c='#d3d3d3')
            plt.annotate(forplot, xy=(0.0, 0.965), xytext=(0, 0), xycoords='axes fraction', textcoords='offset points', color='lightgray', fontweight='bold', fontsize='12',
            horizontalalignment='left', verticalalignment='center')

            ax1.plot(x_axis, list_meanByLevsa, "b-", label="Mean ("+omflaga+")")
            ax1.plot(x_axis, list_meanByLevsa, "bo", label="Mean ("+omflaga+")")
            ax1.set_xlabel('Date (DayHour)', fontsize=10)
            # Make the y-axis label, ticks and tick labels match the line color.
            ax1.set_ylim(vminOMA, vmaxOMA)
            ax1.set_ylabel('Mean ('+omflaga+')', color='b', fontsize=10)
            ax1.tick_params('y', colors='b')
            plt.xticks(x_axis, DayHour)
            major_ticks = [ DayHour.index(dh) for dh in filter(None,DayHour) ]
            ax1.set_xticks(major_ticks)
            plt.axhline(y=np.mean(list_meanByLevsa),ls='dotted',c='blue')
            
            ax2 = ax1.twinx()
            ax2.plot(x_axis, std_finala, "r-", label="Std. Deviation ("+omflaga+")")
            ax2.plot(x_axis, std_finala, "rs", label="Std. Deviation ("+omflaga+")")
            ax2.set_ylim(vminSTD, vmaxSTD)
            ax2.set_ylabel('Std. Deviation ('+omflaga+')', color='r', fontsize=10)
            ax2.tick_params('y', colors='r')
            plt.axhline(y=np.mean(std_finala),ls='dotted',c='red')

            ax3 = ax1.twinx()
            ax3.plot(x_axis, count_finala, "g-", label="Total Observations"+" ("+cmaski+")")
            ax3.plot(x_axis, count_finala, "g^", label="Total Observations"+" ("+cmaski+")")
            ax3.set_ylim(0, 1.2*np.max(count_finala))
            ax3.set_ylabel('Total Observations'+" ("+cmaski+")", color='g', fontsize=10)
            ax3.tick_params('y', colors='g')
            ax3.spines["right"].set_position(("axes", 1.15))
            plt.yticks(rotation=90)
            plt.axhline(y=np.mean(count_finala),ls='dotted',c='green')

            ax3.set_title(instrument_title, loc='left', fontsize=10)
            ax3.set_title(date_title, loc='right', fontsize=10)

            plt.xticks(x_axis, DayHour)
            major_ticks = [ DayHour.index(dh) for dh in filter(None,DayHour) ]
            ax3.set_xticks(major_ticks)
            plt.title(instrument_title, loc='left', fontsize=9)
            plt.title(date_title, loc='right', fontsize=9)
            plt.subplots_adjust(left=None, bottom=None, right=0.80, top=None)
            plt.tight_layout()
            plt.savefig('time_series_'+str(varName) + '-' + str(varType)+'_'+omflaga+'_'+forplotname+'.png', bbox_inches='tight', dpi=100)
            if Clean:
                plt.clf()

            ##### OMF and OMA

            fig = plt.figure(figsize=(6, 4))
            fig, ax1 = plt.subplots(1, 1)
            plt.style.use('seaborn-v0_8-ticks')

            plt.annotate(forplot, xy=(0.0, 0.965), xytext=(0, 0), xycoords='axes fraction', textcoords='offset points', color='lightgray', fontweight='bold', fontsize='12',
            horizontalalignment='left', verticalalignment='center')

            plt.axhline(y=0.0,ls='solid',c='#d3d3d3')
            ax1.plot(x_axis, list_meanByLevs, "b-", label="Mean ("+omflag+")")
            ax1.plot(x_axis, list_meanByLevs, "bo", label="")
            ax1.set_xlabel('Date (DayHour)', fontsize=10)
            # Make the y-axis label, ticks and tick labels match the line color.
            ax1.set_ylim(vminOMA, vmaxOMA)
            ax1.tick_params('y', colors='b')
            plt.xticks(x_axis, DayHour)
            major_ticks = [ DayHour.index(dh) for dh in filter(None,DayHour) ]
            ax1.set_xticks(major_ticks)
            plt.axhline(y=np.mean(list_meanByLevs),ls='dotted',c='blue')
            
            ax1.plot(x_axis, list_meanByLevsa, "r-", label="Mean ("+omflaga+")")
            ax1.plot(x_axis, list_meanByLevsa, "rs", label="")
            ax1.set_ylim(vminOMA, vmaxOMA)
            ax1.tick_params('y', colors='black')
            plt.axhline(y=np.mean(list_meanByLevsa),ls='dotted',c='red')

            plt.xticks(x_axis, DayHour)
            major_ticks = [ DayHour.index(dh) for dh in filter(None,DayHour) ]
            ax1.set_xticks(major_ticks)
            plt.title(instrument_title, loc='left', fontsize=9)
            plt.title(date_title, loc='right', fontsize=9)
            plt.subplots_adjust(left=None, bottom=None, right=0.80, top=None)

            ybox1 = TextArea('Mean ('+omflag+')' , textprops=dict(color="b", size=12,rotation=90,ha='left',va='bottom'))
            ybox2 = TextArea(' and '             , textprops=dict(color="k", size=12,rotation=90,ha='left',va='bottom'))
            ybox3 = TextArea('Mean ('+omflaga+')', textprops=dict(color="r", size=12,rotation=90,ha='left',va='bottom'))

            ybox = VPacker(children=[ybox3, ybox2, ybox1],align="bottom", pad=0, sep=5)

            anchored_ybox = AnchoredOffsetbox(loc=3, child=ybox, pad=0., frameon=False, bbox_to_anchor=(-0.12, 0.16), 
                                                bbox_transform=ax1.transAxes, borderpad=0.)

            ax1.add_artist(anchored_ybox)
            plt.legend()

            plt.tight_layout()
            plt.savefig('time_series_'+str(varName) + '-' + str(varType)+'_OmFOmA_'+ forplotname +'.png', bbox_inches='tight', dpi=100)

            # OMF and OMA and StdDev

            fig = plt.figure(figsize=(6, 4))
            fig, ax1 = plt.subplots(1, 1)
            plt.style.use('seaborn-v0_8-ticks')
            
            ax1.plot(x_axis, list_meanByLevs, lw=2, label='OmF Mean', color='blue', zorder=1)
            ax1.fill_between(x_axis, OMF_inf, OMF_sup, label='OmF Std Dev',  facecolor='blue', alpha=0.3, zorder=1)
            ax1.plot(x_axis, list_meanByLevsa, lw=2, label='OmA Mean', color='red', zorder=2)
            ax1.fill_between(x_axis, OMA_inf, OMA_sup, label='OmA Std Dev',  facecolor='red', alpha=0.3, zorder=2)
            ybox1 = TextArea(' OmF ' , textprops=dict(color="b", size=12,rotation=90,ha='left',va='bottom'))
            ybox2 = TextArea(' | '             , textprops=dict(color="k", size=12,rotation=90,ha='left',va='bottom'))
            ybox3 = TextArea(' OmA ', textprops=dict(color="r", size=12,rotation=90,ha='left',va='bottom'))

            ybox = VPacker(children=[ybox3, ybox2, ybox1],align="bottom", pad=0, sep=5)

            anchored_ybox = AnchoredOffsetbox(loc=3, child=ybox, pad=0., frameon=False, bbox_to_anchor=(-0.125, 0.42), 
                                                bbox_transform=ax1.transAxes, borderpad=0.)

            ax1.add_artist(anchored_ybox)
            ax1.set_xlabel('Date (DayHour)', fontsize=12)
            ax1.set_ylim(omfoma_limit_inf,omfoma_limit_sup)
            ax1.legend(bbox_to_anchor=(-0.11, -0.25),ncol=4,loc='lower left', fancybox=True, shadow=False, frameon=True, framealpha=1.0, fontsize='11', facecolor='white', edgecolor='lightgray')
            plt.grid(axis='y', color='lightgray', linestyle='-.', linewidth=0.5, zorder=0)

            ax2 = ax1.twinx()
            ax2.plot(x_axis, list_countByLevsa, lw=2, label='OmA', linestyle='--', color='green', zorder=3)
            ax2.plot(x_axis, list_countByLevs, lw=2, label='OmF', linestyle=':', color='purple', zorder=3)
            ax2.set_ylabel('Total Observations (OmF | OmA)'+"\n ("+cmaski+")", fontsize=12)
            ax2.set_ylim(0, (np.max(list_countByLevsa) + np.max(list_countByLevsa)/5))
            ax2.legend(loc='upper left', ncol=2, fancybox=True, shadow=False, frameon=True, framealpha=1.0, fontsize='11', facecolor='white', edgecolor='lightgray')
            
            plt.xticks(x_axis, DayHour)
            major_ticks = [ DayHour.index(dh) for dh in filter(None,DayHour) ]
            ax2.set_xticks(major_ticks)
            plt.title(instrument_title, loc='left', fontsize=10)
            plt.title(date_title, loc='right', fontsize=10)
        
            t = plt.annotate(forplot, xy=(0.78, 0.995), xytext=(-9, -9), xycoords='axes fraction', textcoords='offset points', color='darkgray', fontweight='bold', fontsize='10',
                                horizontalalignment='center', verticalalignment='center')
            t.set_bbox(dict(facecolor='whitesmoke', alpha=1.0, edgecolor='whitesmoke', boxstyle="square,pad=0.3"))

            plt.tight_layout()
            plt.savefig('time_series_'+str(varName) + '-' + str(varType)+'_OmFOmA_StdDev_'+ forplotname +'.png', bbox_inches='tight', dpi=100)

        # Cleaning up
        if Clean:
            plt.close('all')

        print(' Done!')
        print()
        
               

        return
        
# radiance inicio



    def time_series_radi(  # type: ignore[override]  # assinatura mantida (método de classe)
        self,
        varName: Optional[str] = None,
        varType: Optional[str] = None,
        mask: Optional[str] = None,
        dateIni: Optional[str | int] = None,
        dateFin: Optional[str | int] = None,
        nHour: str = "06",
        vminOMA: Optional[float] = None,
        vmaxOMA: Optional[float] = None,
        vminSTD: float | None = 0.0,
        vmaxSTD: float | None = 14.0,
        channel: Optional[int | List[int]] = None,
        Clean: Optional[bool] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Plot time series and Hovmöller panels of radiance OmF/OmA statistics per channel.
    
        This function preserves the legacy behavior/output while making the code
        more robust to missing dates, flexible to different column names
        (``channel`` or ``nchan``), and safer against empty selections.
    
        Parameters
        ----------
        varName : str or None
            Variable/instrument name key in ``self[f].obsInfo`` dict.
        varType : str or None
            Variable type used as `.loc[varType]` selector.
        mask : str or None
            Pandas query-string applied to the DataFrame (e.g., ``"iuse>-99999.9"``).
            If ``None``, uses ``"iuse>-99999.9"`` (i.e., no effective filtering).
        dateIni : str | int | None
            Start date in ``YYYYMMDDHH`` (string or int).
        dateFin : str | int | None
            End date in ``YYYYMMDDHH`` (string or int), inclusive.
        nHour : str, default "06"
            Temporal step in hours between cycles (e.g., "06").
        vminOMA, vmaxOMA : float or None
            Y-limits for mean (OmF/OmA) plots. If both ``None``, computed from data.
        vminSTD, vmaxSTD : float or None
            Y-limits for standard deviation plots. If both ``None``, computed from data.
        channel : int | list[int] | None
            Specific channel or list of channels to plot. If ``None``, the function
            infers channels from the data; if empty, it defaults to ``[1..15]``.
        Clean : bool or None
            If ``True``, clears figures from memory after saving.
            Defaults to ``True`` if not provided.
        *args, **kwargs :
            Extra positional/keyword-arguments are accepted for forward compatibility
            but are currently ignored.
    
        Notes
        -----
        - Expects ``self[f].obsInfo[varName]`` to be a (potentially multi-indexed)
          DataFrame where rows can be selected with ``.loc[varType]`` and then
          filtered via ``.query(mask)``.
        - Accepts either ``channel`` or ``nchan`` column names for the channel id.
        - Missing days/hours are handled by filling with sentinel ``-99`` (masked
          downstream for plotting).
        - The function prints progress and basic summaries (legacy behavior).
    
        Returns
        -------
        None
            Saves PNG figures to the current working directory and prints progress.
        """
    
        # -------------------- setup & helpers --------------------
        def _safe_print(msg: str, color: str = "") -> None:
            """Print with optional legacy `setcolor` support."""
            try:
                if color and "setcolor" in globals():
                    # type: ignore[name-defined]
                    print(getattr(setcolor, color, ""))  # type: ignore[misc]
                    print(msg + getattr(setcolor, "ENDC", ""))  # type: ignore[misc]
                else:
                    print(msg)
            except Exception:
                print(msg)
    
        def _chan_series(df) -> np.ndarray:
            """Return channel series as numpy array of ints (empty if not found)."""
            if df is None:
                return np.array([], dtype=int)
            cols = df.columns
            if "nchan" in cols:
                return np.asarray(df["nchan"]).astype(int, copy=False)
            if "channel" in cols:
                return np.asarray(df["channel"]).astype(int, copy=False)
            return np.array([], dtype=int)
    
        def _get_series(df, name: str) -> np.ndarray:
            """Return numpy array for column `name` or empty array if missing."""
            if df is None or name not in getattr(df, "columns", []):
                return np.array([], dtype=float)
            return np.asarray(df[name]).astype(float, copy=False)
    
        def _ensure_iterable_channels(ch, name: str) -> Tuple[List[int], int, List[int]]:
            """
            Normalize `channel` parameter.
            Returns (zchan_list, chanList_flag, zchans_def).
            """
            # Carol: no caso do atms são 22 canais
            if name == 'amsua':
                total_chan = list(range(1, 16))
            elif name == 'atms':
                total_chan = list(range(1, 23))
            else:
                total_chan = [0] # Carol: não sei o que colocar - precisa melhorar!
                _safe_print(
                    f"    >>> total_chan not defined for {name} sensor in _ensure_iterable_channels <<< ",
                    color="WARNING",
                )
            
            if isinstance(ch, list):
                return [int(c) for c in ch], 1, [int(c) for c in ch]
            if ch is None:
                # AMSU-A typical default 1..15 kept for legacy behavior
                return total_chan, 0, total_chan
            # single int
            return [int(ch)], 0, total_chan
    
        # -------------------- defaults & inputs --------------------
        Clean = True if Clean is None else bool(Clean)
    
        maski = "iuse>-99999.9" if mask is None else str(mask)
        cmaski = "iuse = All" if mask is None else str(mask)
    
        # instrument info (legacy print)
        try:
            varInfo = getVarInfo(varType, varName, "instrument")  # type: ignore[name-defined]
        except Exception:
            varInfo = None
        varInfo = varInfo if varInfo is not None else "Unknown instrument"
    
        omflag, omflaga = "OmF", "OmA"
        separator = " " + "=" * 109
    
        print()
        print(separator)
        print(f" Variable: {varName}  ||  type: {varType}  ||  {varInfo}  ||  check: {omflag}")
        print(separator)
        print()
    
        # channels normalization
        zchan, chanList, zchans_def = _ensure_iterable_channels(channel,varName) # Carol: problema atms
    
        # dates
        datei = datetime.strptime(str(dateIni), "%Y%m%d%H")
        datef = datetime.strptime(str(dateFin), "%Y%m%d%H")
        hour_step = int(nHour)
        dates: List[datetime] = []
        d = datei
        while d <= datef:
            dates.append(d)
            d += timedelta(hours=hour_step)
    
        # axis labels (DayHour) with 1/4 spacing of ticks
        dayhour_all = [dt.strftime("%d%H") for dt in dates]
        if len(dayhour_all) > 4:
            step = max(1, len(dayhour_all) // 4)
            DayHour = [val if (ix % step) == 0 else "" for ix, val in enumerate(dayhour_all)]
        else:
            DayHour = dayhour_all
    
        # pass 1: detect data presence and collect channels
        info_check: Dict[str, bool] = {}
        found_channels: set[int] = set()
        for f, dt in enumerate(dates):
            key = dt.strftime("%d%H")
            try:
                df = self[f].obsInfo[varName].query(maski).loc[varType]  # type: ignore[index]
            except Exception:
                _safe_print(
                    f"    >>> No information on this date ({dt.strftime('%Y-%m-%d:%H')}) <<< ",
                    color="WARNING",
                )
                info_check[key] = False
                continue
    
            chs = _chan_series(df)
            print('chs = ', chs)
            if chs.size:
                info_check[key] = True
                if channel is None or chanList == 1:
                    # aggregate any present channels
                    # np.unique is faster and avoids repeated set ops
                    found_channels.update(np.unique(chs).tolist())
                    print('found channels = ', found_channels)
                    print('')
                print(dt.strftime(" Preparing data for: Canais de radiancia %Y-%m-%d:%H"))
            else:
                info_check[key] = False
                _safe_print(
                    dt.strftime(" Preparing data for: %Y-%m-%d:%H - No information on this date "),
                    color="WARNING",
                )
    
        # Carol: a variável levs precisa ser igual a zchan se não o eixo y (zlevs) fica errado.
        # Por exemplo: levs = found_channels = [8, 9, 10] como zlevs = [1, 2, ..., 15] 
        # o resultado do canal 8 estava sendo plotado na linha do canal 1 no diagrama. Por isso comentei a linha 1825 e inseri a 1826
        # final channel list
        if found_channels:
            #levs = sorted(int(c) for c in found_channels)
            levs = zchan
        else:
            # if nothing found, keep default definition
            levs = sorted(set(zchans_def))
        
        print('levs = ', levs)
        # labels printed on Y for Hovmöller
        if channel is None or chanList == 1:
            zlevs = [z if z in zchans_def else "" for z in sorted(set(levs + zchans_def))]
        else:
            zlevs = [int(zchan[0])]  # single-channel mode uses 1 column
    
        print('zlevs = ', zlevs)
        print()
        print(separator)
        print()
        print("channels = ", levs)
    
        # containers
        list_meanByLevs: List[Any] = []
        list_stdByLevs: List[Any] = []
        list_countByLevs: List[Any] = []
        list_meanByLevsa: List[Any] = []
        list_stdByLevsa: List[Any] = []
        list_countByLevsa: List[Any] = []
    
        # pass 2: compute stats per date/channel
        for f, dt in enumerate(dates):
            print(dt.strftime(" Calculating for %Y-%m-%d:%H"))
            key = dt.strftime("%d%H")
    
            # prepare dicts for this date
            mean_f: Dict[int, float] = {int(lv): -99.0 for lv in levs}
            std_f: Dict[int, float] = {int(lv): -99.0 for lv in levs}
            cnt_f: Dict[int, int] = {int(lv): -99 for lv in levs}
    
            mean_a: Dict[int, float] = {int(lv): -99.0 for lv in levs}
            std_a: Dict[int, float] = {int(lv): -99.0 for lv in levs}
            cnt_a: Dict[int, int] = {int(lv): -99 for lv in levs}
    
            if info_check.get(key, False):
                try:
                    df = self[f].obsInfo[varName].query(maski).loc[varType]  # type: ignore[index]
                    chs = _chan_series(df)
                    omf = _get_series(df, "omf")
                    oma = _get_series(df, "oma")
    
                    # quick exit if no usable columns
                    if chs.size == 0 or (omf.size == 0 and oma.size == 0):
                        pass
                    else:
                        # channel filter: all or specific
                        if channel is not None and chanList != 1:
                            # single channel path (zchan[0] guaranteed from normalization)
                            c = int(zchan[0])
                            sel = (chs == c)
                            if omf.size:
                                arr = omf[sel]
                                arr = arr[np.isfinite(arr)]
                                if arr.size:
                                    mean_f[c] = float(np.nanmean(arr))
                                    std_f[c] = float(np.nanstd(arr))
                                    cnt_f[c] = int(arr.size)
                            if oma.size:
                                arr = oma[sel]
                                arr = arr[np.isfinite(arr)]
                                if arr.size:
                                    mean_a[c] = float(np.nanmean(arr))
                                    std_a[c] = float(np.nanstd(arr))
                                    cnt_a[c] = int(arr.size)
                        else:
                            # all channels present in `levs`
                            # compute once: channel->indices
                            # speeds up when `levs` is long
                            for c in levs:
                                sel = (chs == int(c))
                                if omf.size:
                                    arr = omf[sel]
                                    arr = arr[np.isfinite(arr)]
                                    if arr.size:
                                        mean_f[c] = float(np.nanmean(arr))
                                        std_f[c] = float(np.nanstd(arr))
                                        cnt_f[c] = int(arr.size)
                                if oma.size:
                                    arr = oma[sel]
                                    arr = arr[np.isfinite(arr)]
                                    if arr.size:
                                        mean_a[c] = float(np.nanmean(arr))
                                        std_a[c] = float(np.nanstd(arr))
                                        cnt_a[c] = int(arr.size)
    
                except Exception:
                    # keep -99 sentinels
                    pass
    
            # append in the expected shape
            if channel is None or chanList == 1:
                # matrix (dates x channels), reversed for legacy orientation
                rev = list(reversed(levs))
                list_meanByLevs.append([mean_f[c] for c in rev])
                list_stdByLevs.append([std_f[c] for c in rev])
                list_countByLevs.append([cnt_f[c] for c in rev])
    
                list_meanByLevsa.append([mean_a[c] for c in rev])
                list_stdByLevsa.append([std_a[c] for c in rev])
                list_countByLevsa.append([cnt_a[c] for c in rev])
            else:
                c = int(zchan[0])
                list_meanByLevs.append(mean_f[c])
                list_stdByLevs.append(std_f[c])
                list_countByLevs.append(cnt_f[c])
    
                list_meanByLevsa.append(mean_a[c])
                list_stdByLevsa.append(std_a[c])
                list_countByLevsa.append(cnt_a[c])
    
        # -------------------- plotting --------------------
        print()
        print(separator)
        print()
        print(" Making Graphics...")
    
        x_axis = np.arange(0, len(DayHour), 1)
    
        # to arrays
        if channel is None or chanList == 1:
            A = np.array(list_meanByLevs, dtype=float)
            S = np.array(list_stdByLevs, dtype=float)
            C = np.array(list_countByLevs, dtype=float)
            Aa = np.array(list_meanByLevsa, dtype=float)
            Sa = np.array(list_stdByLevsa, dtype=float)
            Ca = np.array(list_countByLevsa, dtype=float)
        else:
            # (dates x 1) to reuse same logic
            A = np.array(list_meanByLevs, dtype=float)[:, None]
            S = np.array(list_stdByLevs, dtype=float)[:, None]
            C = np.array(list_countByLevs, dtype=float)[:, None]
            Aa = np.array(list_meanByLevsa, dtype=float)[:, None]
            Sa = np.array(list_stdByLevsa, dtype=float)[:, None]
            Ca = np.array(list_countByLevsa, dtype=float)[:, None]
    
        # masked versions for plotting
        mean_final = np.ma.masked_where(A == -99, A)
        std_final = np.ma.masked_where(S == -99, S)
        count_final = np.ma.masked_where(C == -99, C)
    
        mean_finala = np.ma.masked_where(Aa == -99, Aa)
        std_finala = np.ma.masked_where(Sa == -99, Sa)
        count_finala = np.ma.masked_where(Ca == -99, Ca)
    
        OMF_inf, OMF_sup = A - S, A + S
        OMA_inf, OMA_sup = Aa - Sa, Aa + Sa
    
        # auto-limits
        def _finite_min(x: np.ndarray, default: float = 0.0) -> float:
            with np.errstate(invalid="ignore"):
                m = np.nanmin(x) if x.size else np.nan
            return float(m) if np.isfinite(m) else default
    
        def _finite_max(x: np.ndarray, default: float = 1.0) -> float:
            with np.errstate(invalid="ignore"):
                m = np.nanmax(x) if x.size else np.nan
            return float(m) if np.isfinite(m) else default
    
        mean_limit_inf = _finite_min(np.asarray([mean_final.min(), mean_finala.min()], dtype=float), 0.0)
        mean_limit_sup = _finite_max(np.asarray([mean_final.max(), mean_finala.max()], dtype=float), 1.0)
        std_limit_inf = _finite_min(np.asarray([std_final.min(), std_finala.min()], dtype=float), 0.0)
        std_limit_sup = _finite_max(np.asarray([std_final.max(), std_finala.max()], dtype=float), 1.0)
    
        omfoma_limit_inf = _finite_min(np.asarray([OMF_inf.min(), OMA_inf.min()], dtype=float), 0.0)
        omfoma_limit_sup = _finite_max(np.asarray([OMF_sup.max(), OMA_sup.max()], dtype=float), 1.0)
        omfoma_limit_inf = 0.9 * omfoma_limit_inf if omfoma_limit_inf > 0 else 1.1 * omfoma_limit_inf
        omfoma_limit_sup = 1.1 * omfoma_limit_sup
    
        if vminOMA is None and vmaxOMA is None:
            vminOMA, vmaxOMA = mean_limit_inf, 1.1 * mean_limit_sup
        vminOMA = 0.9 * vminOMA if float(vminOMA) > 0 else 1.1 * float(vminOMA)
        vmaxOMAabs = max(abs(float(vminOMA)), abs(float(vminOMA)))  # keeps legacy “sym abs” intent
    
        if vminSTD is None and vmaxSTD is None:
            vminSTD, vmaxSTD = std_limit_inf - 0.1 * std_limit_inf, 1.1 * std_limit_sup
    
        date_title = f"{datei.strftime('%d%b')}-{dates[-1].strftime('%d%b')} {dates[-1].strftime('%Y')}"
        instrument_title = f"{varName}-{varType}  |  {varInfo}"
    
        # ---- multi-channel Hovmöller plots ----
        if channel is None or chanList == 1:
            y_ticks = np.arange(0, len(zlevs), 1)
            x_ticks = np.arange(0, len(DayHour), 1)
    
            def _common_panel(ax, arr, vmin, vmax, cmap, title_left: str, title_right: str, ylabel: str, xlabel: str) -> None:
                ax.add_patch(
                    mpl.patches.Rectangle(
                        (-1, -1),
                        (len(DayHour) + 1),
                        (len(levs) + 3),
                        hatch="xxxxx",
                        color="black",
                        fill=False,
                        snap=False,
                        zorder=0,
                    )
                )
                im = ax.imshow(
                    np.flipud(arr.T),
                    origin="lower",
                    vmin=vmin,
                    vmax=vmax,
                    cmap=cmap,
                    aspect="auto",
                    zorder=1,
                    interpolation="none",
                )
                plt.colorbar(im, orientation="horizontal", pad=0.18, shrink=1.0)
                ax.set_title(title_left, loc="left", fontsize=10)
                ax.set_title(title_right, loc="right", fontsize=10)
                ax.set_xlabel(xlabel) # carol -> inseri, estava faltando!
                ax.set_ylabel(ylabel)
                ax.set_xticks(x_ticks)
                ax.set_yticks(y_ticks)
                ax.set_xticklabels(DayHour)
                ax.set_yticklabels(zlevs)
                major_ticks = [DayHour.index(dh) for dh in filter(None, DayHour)]
                ax.set_xticks(major_ticks)
    
            plt.rcParams["axes.facecolor"] = "None"
            plt.rcParams["hatch.linewidth"] = 0.3
    
            # OmF
            fig = plt.figure(figsize=(6, 9))
            ax1 = plt.subplot(3, 1, 1)
            _common_panel(ax1, mean_final, -vmaxOMAabs, vmaxOMAabs, "seismic", instrument_title, date_title, "Channels", f"Mean ({omflag})")
            plt.xlabel('Mean ('+omflag+')', labelpad=50) # carol -> inseri
    
            ax2 = plt.subplot(3, 1, 2)
            _common_panel(ax2, std_final, float(vminSTD), float(vmaxSTD), "Blues", instrument_title, date_title, "Channels", f"Standard Deviation ({omflag})")
            plt.xlabel('Standard Deviation ('+omflag+')', labelpad=50) # carol -> inseri
    
            ax3 = plt.subplot(3, 1, 3)
            vmax_cnt = _finite_max(np.asarray([count_final.max()], dtype=float), 1.0)
            _common_panel(ax3, count_final, 0.0, vmax_cnt, "gist_heat_r", instrument_title, date_title, "Channels", f"Total Observations ({cmaski})")
            plt.xlabel('Total Observations'+" ("+cmaski+")", labelpad=50) # carol -> inseri
    
            plt.tight_layout()
            plt.savefig(f"hovmoller_{varName}-{varType}_{omflag}.png", bbox_inches="tight", dpi=100)
            if Clean:
                plt.clf()
    
            # OmA
            fig = plt.figure(figsize=(6, 9))
            ax1 = plt.subplot(3, 1, 1)
            _common_panel(ax1, mean_finala, -vmaxOMAabs, vmaxOMAabs, "seismic", instrument_title, date_title, "Channels", f"Mean ({omflaga})")
            plt.xlabel('Mean ('+omflaga+')', labelpad=50) # carol -> inseri
    
            ax2 = plt.subplot(3, 1, 2)
            _common_panel(ax2, std_finala, float(vminSTD), float(vmaxSTD), "Blues", instrument_title, date_title, "Channels", f"Standard Deviation ({omflaga})")
            plt.xlabel('Standard Deviation ('+omflaga+')', labelpad=50) # carol -> inseri
    
            ax3 = plt.subplot(3, 1, 3)
            vmax_cnta = _finite_max(np.asarray([count_finala.max()], dtype=float), 1.0)
            _common_panel(ax3, count_finala, 0.0, vmax_cnta, "gist_heat_r", instrument_title, date_title, "Channels", f"Total Observations ({cmaski})")
            plt.xlabel('Total Observations'+" ("+cmaski+")", labelpad=50) # carol -> inseri
    
            plt.tight_layout()
            plt.savefig(f"hovmoller_{varName}-{varType}_{omflaga}.png", bbox_inches="tight", dpi=100)
            if Clean:
                plt.clf()
    
        # ---- single-channel time series panels ----
        else:
            # dados 1D já prontos (listas) + envelopes
            plt.style.use("seaborn-v0_8-ticks")
            c = int(zchan[0])
            forplot = f"Channel = {c}"
            forplotname = f"Channel_{c}"
    
            # helper to place major xticks only where DayHour has labels
            def _apply_xticks(ax):
                ax.set_xticks(x_axis)
                ax.set_xticklabels(DayHour)
                major_ticks = [DayHour.index(dh) for dh in filter(None, DayHour)]
                ax.set_xticks(major_ticks)
    
            # Mean & Std (OmF)
            fig, ax1 = plt.subplots(1, 1, figsize=(6, 4))
            ax1.axhline(y=0.0, ls="solid", c="#d3d3d3")
            ax1.plot(x_axis, list_meanByLevs, "b-", label=f"Mean ({omflag})")
            ax1.plot(x_axis, list_meanByLevs, "bo", label="")
            ax1.set_xlabel("Date (DayHour)", fontsize=10)
            ax1.set_ylim(float(vminOMA), float(vmaxOMA))
            ax1.set_ylabel(f"Mean ({omflag})", color="b", fontsize=10)
            ax1.tick_params("y", colors="b")
            _apply_xticks(ax1)
            if len(list_meanByLevs):
                ax1.axhline(y=float(np.nanmean(list_meanByLevs)), ls="dotted", c="blue")
    
            ax2 = ax1.twinx()
            ax2.plot(x_axis, np.squeeze(S), "r-", label=f"Std. Deviation ({omflag})")
            ax2.plot(x_axis, np.squeeze(S), "rs", label="")
            ax2.set_ylim(float(vminSTD), float(vmaxSTD))
            ax2.set_ylabel(f"Std. Deviation ({omflag})", color="r", fontsize=10)
            ax2.tick_params("y", colors="r")
            if np.size(S):
                ax2.axhline(y=float(np.nanmean(S)), ls="dotted", c="red")
    
            ax3 = ax1.twinx()
            ax3.plot(x_axis, np.squeeze(C), "g-", label=f"Total Observations ({cmaski})")
            ax3.plot(x_axis, np.squeeze(C), "g^", label="")
            vmax_cnt = _finite_max(np.asarray([np.nanmax(C)], dtype=float), 1.0)
            ax3.set_ylim(0.0, vmax_cnt + (vmax_cnt / 8.0))
            ax3.set_ylabel(f"Total Observations ({cmaski})", color="g", fontsize=10)
            ax3.tick_params("y", colors="g")
            ax3.spines["right"].set_position(("axes", 1.15))
            ax3.set_title(instrument_title, loc="left", fontsize=10)
            ax3.set_title(date_title, loc="right", fontsize=10)
            _apply_xticks(ax3)
    
            plt.title(instrument_title, loc="left", fontsize=9)
            plt.title(date_title, loc="right", fontsize=9)
            plt.tight_layout()
            plt.savefig(f"time_series_{varName}-{varType}_{omflag}_{forplotname}.png", bbox_inches="tight", dpi=100)
            if Clean:
                plt.clf()
    
            # Mean & Std (OmA)
            fig, ax1 = plt.subplots(1, 1, figsize=(6, 4))
            ax1.axhline(y=0.0, ls="solid", c="#d3d3d3")
            ax1.plot(x_axis, list_meanByLevsa, "b-", label=f"Mean ({omflaga})")
            ax1.plot(x_axis, list_meanByLevsa, "bo", label="")
            ax1.set_xlabel("Date (DayHour)", fontsize=10)
            ax1.set_ylim(float(vminOMA), float(vmaxOMA))
            ax1.set_ylabel(f"Mean ({omflaga})", color="b", fontsize=10)
            ax1.tick_params("y", colors="b")
            _apply_xticks(ax1)
            if len(list_meanByLevsa):
                ax1.axhline(y=float(np.nanmean(list_meanByLevsa)), ls="dotted", c="blue")
    
            ax2 = ax1.twinx()
            ax2.plot(x_axis, np.squeeze(Sa), "r-", label=f"Std. Deviation ({omflaga})")
            ax2.plot(x_axis, np.squeeze(Sa), "rs", label="")
            ax2.set_ylim(float(vminSTD), float(vmaxSTD))
            ax2.set_ylabel(f"Std. Deviation ({omflaga})", color="r", fontsize=10)
            ax2.tick_params("y", colors="r")
            if np.size(Sa):
                ax2.axhline(y=float(np.nanmean(Sa)), ls="dotted", c="red")
    
            ax3 = ax1.twinx()
            ax3.plot(x_axis, np.squeeze(Ca), "g-", label=f"Total Observations ({cmaski})")
            ax3.plot(x_axis, np.squeeze(Ca), "g^", label="")
            vmax_cnta = _finite_max(np.asarray([np.nanmax(Ca)], dtype=float), 1.0)
            ax3.set_ylim(0.0, 1.2 * vmax_cnta)
            ax3.set_ylabel(f"Total Observations ({cmaski})", color="g", fontsize=10)
            ax3.tick_params("y", colors="g")
            ax3.spines["right"].set_position(("axes", 1.15))
            ax3.set_title(instrument_title, loc="left", fontsize=10)
            ax3.set_title(date_title, loc="right", fontsize=10)
            _apply_xticks(ax3)
    
            plt.title(instrument_title, loc="left", fontsize=9)
            plt.title(date_title, loc="right", fontsize=9)
            plt.tight_layout()
            plt.savefig(f"time_series_{varName}-{varType}_{omflaga}_{forplotname}.png", bbox_inches="tight", dpi=100)
            if Clean:
                plt.clf()
    
            # OmF & OmA overlay
            from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker
    
            fig, ax1 = plt.subplots(1, 1, figsize=(6, 4))
            ax1.axhline(y=0.0, ls="solid", c="#d3d3d3")
            ax1.plot(x_axis, list_meanByLevs, "b-", label=f"Mean ({omflag})")
            ax1.plot(x_axis, list_meanByLevsa, "r-", label=f"Mean ({omflaga})")
            ax1.set_xlabel("Date (DayHour)", fontsize=10)
            ax1.set_ylim(float(vminOMA), float(vmaxOMA))
            _apply_xticks(ax1)
            if len(list_meanByLevs):
                ax1.axhline(y=float(np.nanmean(list_meanByLevs)), ls="dotted", c="blue")
            if len(list_meanByLevsa):
                ax1.axhline(y=float(np.nanmean(list_meanByLevsa)), ls="dotted", c="red")
    
            ybox1 = TextArea(f"Mean ({omflag})", textprops=dict(color="b", size=12, rotation=90, ha="left", va="bottom"))
            ybox2 = TextArea(" and ", textprops=dict(color="k", size=12, rotation=90, ha="left", va="bottom"))
            ybox3 = TextArea(f"Mean ({omflaga})", textprops=dict(color="r", size=12, rotation=90, ha="left", va="bottom"))
            ybox = VPacker(children=[ybox3, ybox2, ybox1], align="bottom", pad=0, sep=5)
            anchored_ybox = AnchoredOffsetbox(loc=3, child=ybox, pad=0.0, frameon=False, bbox_to_anchor=(-0.12, 0.16), bbox_transform=ax1.transAxes, borderpad=0.0)
            ax1.add_artist(anchored_ybox)
    
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"time_series_{varName}-{varType}_OmFOmA_{forplotname}.png", bbox_inches="tight", dpi=100)
            if Clean:
                plt.clf()
    
            # OmF & OmA with Std envelopes
            fig, ax1 = plt.subplots(1, 1, figsize=(6, 4))
            ax1.plot(x_axis, list_meanByLevs, lw=2, label="OmF Mean", color="blue", zorder=1)
            ax1.fill_between(x_axis, (A - S).flatten(), (A + S).flatten(), label="OmF Std Dev", alpha=0.3, zorder=1)
            ax1.plot(x_axis, list_meanByLevsa, lw=2, label="OmA Mean", color="red", zorder=2)
            ax1.fill_between(x_axis, (Aa - Sa).flatten(), (Aa + Sa).flatten(), label="OmA Std Dev", alpha=0.3, zorder=2)
    
            ybox1 = TextArea(" OmF ", textprops=dict(color="b", size=12, rotation=90, ha="left", va="bottom"))
            ybox2 = TextArea(" | ", textprops=dict(color="k", size=12, rotation=90, ha="left", va="bottom"))
            ybox3 = TextArea(" OmA ", textprops=dict(color="r", size=12, rotation=90, ha="left", va="bottom"))
            ybox = VPacker(children=[ybox3, ybox2, ybox1], align="bottom", pad=0, sep=5)
            anchored_ybox = AnchoredOffsetbox(loc=3, child=ybox, pad=0.0, frameon=False, bbox_to_anchor=(-0.125, 0.42), bbox_transform=ax1.transAxes, borderpad=0.0)
            ax1.add_artist(anchored_ybox)
    
            ax1.set_xlabel("Date (DayHour)", fontsize=12)
            ax1.set_ylim(omfoma_limit_inf, omfoma_limit_sup)
            ax1.legend(
                bbox_to_anchor=(-0.11, -0.25),
                ncol=4,
                loc="lower left",
                fancybox=True,
                shadow=False,
                frameon=True,
                framealpha=1.0,
                fontsize="11",
            )
            plt.grid(axis="y", color="lightgray", linestyle="-.", linewidth=0.5, zorder=0)
    
            ax2 = ax1.twinx()
            ax2.plot(x_axis, list_countByLevsa, lw=2, label="OmA", linestyle="--", color="green", zorder=3)
            ax2.plot(x_axis, list_countByLevs, lw=2, label="OmF", linestyle=":", color="purple", zorder=3)
            ax2.set_ylabel(f"Total Observations (OmF | OmA)\n ({cmaski})", fontsize=12)
            ymax_ = max(
                1.0,
                float(np.nanmax(list_countByLevsa)) if len(list_countByLevsa) else 1.0,
            )
            ax2.set_ylim(0.0, ymax_ + ymax_ / 5.0)
            ax2.legend(loc="upper left", ncol=2, fancybox=True, shadow=False, frameon=True, framealpha=1.0, fontsize="11")
    
            ax2.set_xticks(x_axis)
            ax2.set_xticklabels(DayHour)
            major_ticks = [DayHour.index(dh) for dh in filter(None, DayHour)]
            ax2.set_xticks(major_ticks)
            plt.title(instrument_title, loc="left", fontsize=10)
            plt.title(date_title, loc="right", fontsize=10)
    
            plt.tight_layout()
            plt.savefig(f"time_series_{varName}-{varType}_OmFOmA_StdDev_{forplotname}.png", bbox_inches="tight", dpi=100)
    
        # cleanup
        if Clean:
            plt.close("all")
    
        print(" Done!\n")
        return






# radiance final
    def statcount(self, varName=None, varType=None, noiqc=False,
                  dateIni=None, dateFin=None, nHour="06",
                  channel=None, figTS=False, figMap=False, **kwargs):
        '''
        Conta observações por data (e opcionalmente por canal) para radiância.
        Assinatura original preservada. Robusto a datas sem dados.
    
        Parâmetros-chave
        ----------------
        varName : str   (ex.: 'amsua')
        varType : str   (ex.: 'n19')
        noiqc   : bool  (True => ignora qc: idqc==0 não é imposto)
        dateIni/dateFin: int YYYYMMDDHH
        nHour   : str   ("06", "12", ...)
        channel : None => todos os canais | int => apenas esse canal
        figTS   : bool  (gera série temporal)
        figMap  : bool  (mantido por compat; aqui não plota mapa base se não houver cartopy)


        The StatCount function plots a time series of assimilated, monitored and rejected data. 

        Example:

        varName = 'uv'           # Variable
        varType = 224            # Source Type
        noiqc = False            # noiqc GSI namelist parameter (OI QC - True or False)
        dateIni = 2013010100     # Inicial Date
        dateFin = 2013010900     # Final Date
        nHour = "06"             # Time Interval
        channel = None           # Radiance channel number (None for the conventional dataset)
        figTS = True             # Creates the time series plot
        figMap = False           # Creates the spatial plot for each time
        
        ! Case conventional dataset: channel = None
        ! The QC process creates a number indicating the data quality for each observation.
        ! These numbers are called QC markers in PrepBUFR files and are important as parts of
        ! the observation information. GSI uses QC markers to decide how to use the data. A 
        ! brief summary of the meaning of the QC markers is as follows:
        ! 
        !    +-----------------+-----------------------------------------------------------+
        !    | QC markes range | Data Process in GSI                                       |
        !    +-----------------+-----------------------------------------------------------+
        !    |  > 15 or        |GSI skips these observations during reading procedure. That|
        !    |  <= 0           |means these observations are tossed                        | 
        !    +-----------------+-----------------------------------------------------------+
        !    |  >= lim_qm      |These observations will be in monitoring status. That means|
        !    |  and            |these observations will be read in and be processed through|
        !    |  < = 15         |GSI QC process (gross check) and innovation calculation    | 
        !    |                 |stage but will not be used in inner iteration.             |
        !    +-----------------+-----------------------------------------------------------+
        !    |  > 0            |Observations will be used in further gross check (failure  |
        !    |  and            |observation will be list in rejection), innovation         |
        !    |  < lim_qm       |caalculation, and the analysis (inner iteration).          |
        !    +-----------------+-----------------------------------------------------------+

        !    +----------------------+---------------+---------------+
        !    |The value of namelist | lim_qm for Ps | lim_qm others |
        !    |option noiqc          |               |               |
        !    +----------------------+---------------+---------------+
        !    |True (without OI QC)  |       7       |       8       |
        !    +----------------------+---------------+---------------+
        !    |False (with OI QC)    |       4       |       4       |
        !    +----------------------+---------------+---------------+
        
        
        ! Case radiance dataset: channel = number
        ! There are three types of data classification: assimilated, monitored and rejected.
        ! Monitored data is organized into two groups: possibly assimilated and possibly rejected.
        !
        !    +------------------------+-------------+--------------------+
        !    |                        |   idqc      |        iuse        |
        !    +------------------------+-------------+--------------------+
        !    | Assimilated            |   == 0      |   >= 1             |
        !    +------------------------+-------------+--------------------+
        !    |            assimilated |   == 0      |   >= -1 and < 1    |
        !    | Monitored              |             |                    |
        !    |            rejected    |   != 0      |   >= -1 and < 1    |
        !    +------------------------+-------------+--------------------+
        !    | Rejected               |   != 0      |   >= 1             |
        !    +------------------------+-------------+--------------------+
        '''


    
        # ---------- helpers ----------
        def _chan_series(df):
            if "nchan" in df.columns:
                return df["nchan"]
            if "channel" in df.columns:
                return df["channel"]
            return pd.Series([], dtype="int64")
    
        def _safe_len(x):
            try: return int(len(x))
            except Exception: return 0
    
        # máscara “iuse” e “idqc” (quando noiqc=False, aplica idqc==0)
        base_mask = "iuse>-99999.9"
        if not noiqc:
            base_mask += " & idqc==0"
    
        # cabeçalho
        sep = " " + "="*109
        print("\n"+sep)
        instr = getVarInfo(varType, varName, "instrument") or "Unknown instrument"
        print(f" Variable: {varName}  ||  type: {varType}  ||  {instr}  ||  STATCOUNT")
        print(sep+"\n")
    
        # parse datas
        datei = datetime.strptime(str(dateIni), "%Y%m%d%H")
        datef = datetime.strptime(str(dateFin), "%Y%m%d%H")
        step  = int(nHour)
    
        # ---------- 1) Varre datas: detecta dados e coleta canais ----------
        DayHour, info_check = [], {}
        chans_seen = set()
        f = 0
        d = datei
        while d <= datef:
            dh = d.strftime("%d%H")
            DayHour.append(dh)
            try:
                df_t = self[f].obsInfo[varName].query(base_mask).loc[varType]
                if _safe_len(df_t) > 0:
                    info_check[dh] = True
                    chs = _chan_series(df_t)
                    if _safe_len(chs) > 0:
                        chans_seen.update(int(c) for c in np.asarray(chs).astype(int))
                else:
                    info_check[dh] = False
                    print("++++++++++++++++++++++++++ ERROR: file reading --> STATCOUNT ++++++++++++++++++++++++++")
                    print(setcolor.WARNING + f"    >>> No information on this date ({d.strftime('%Y-%m-%d:%H')}) <<< " + setcolor.ENDC)
            except Exception:
                info_check[dh] = False
                print("++++++++++++++++++++++++++ ERROR: file reading --> STATCOUNT ++++++++++++++++++++++++++")
                print(setcolor.WARNING + f"    >>> No information on this date ({d.strftime('%Y-%m-%d:%H')}) <<< " + setcolor.ENDC)
            finally:
                f += 1
                d += timedelta(hours=step)
    
        # modo de canal / rótulos
        if channel is None:
            chan_mode = "all"
            forplot = "All Channels"
            forplotname = "All_Channels"
        else:
            chan_mode = "one"
            channel   = int(channel)
            forplot = f"Channel = {channel}"
            forplotname = f"Channel_{channel}"
    
        chans_sorted = sorted(chans_seen) if chans_seen else []
    
        # ---------- 2) Contagem por data ----------
        total_counts = []
        per_channel_counts = {int(c): [] for c in chans_sorted}
        f = 0
        d = datei
        while d <= datef:
            dh = d.strftime("%d%H")
            tot = 0
            per_chan = {int(c): 0 for c in chans_sorted}
            try:
                if info_check.get(dh, False):
                    df_t = self[f].obsInfo[varName].query(base_mask).loc[varType]
                    chs = _chan_series(df_t)
                    if chan_mode == "one":
                        if _safe_len(chs) > 0:
                            arr = np.asarray(chs).astype(int)
                            tot = int(np.count_nonzero(arr == channel))
                        else:
                            tot = 0
                    else:
                        tot = _safe_len(df_t)
                        if _safe_len(chs) > 0:
                            arr = np.asarray(chs).astype(int)
                            for c in chans_sorted:
                                per_chan[c] = int(np.count_nonzero(arr == c))
                # avança f correspondente a esta data
                f += 1
            except Exception:
                f += 1  # não trava sequência
            total_counts.append(tot)
            for c in chans_sorted:
                per_channel_counts[c].append(per_chan[c])
            d += timedelta(hours=step)
    
        # ---------- 3) Plots ----------
        # ticks a cada ~1/4
        x = np.arange(len(DayHour))
        if len(DayHour) > 4:
            ticks = [i for i in range(len(DayHour)) if i % max(1, len(DayHour)//4) == 0]
        else:
            ticks = list(range(len(DayHour)))
    
        # Série temporal (figTS)
        if figTS:
            plt.style.use('seaborn-v0_8-ticks')
            if chan_mode == "one":
                fig, ax = plt.subplots(1, 1, figsize=(7, 4))
                ax.plot(x, total_counts, "-o")
                ax.set_ylabel("Total Observations (" + ("no IQC" if noiqc else "iuse=idqc==0") + ")")
                ax.set_xlabel("Date (DayHour)")
                ax.set_xticks(ticks); ax.set_xticklabels([DayHour[i] for i in ticks])
                ax.set_title(f"{varName}-{varType}  |  {instr}", loc="left", fontsize=10)
                ax.set_title(f"{datei.strftime('%d%b')}-{datef.strftime('%d%b %Y')}", loc="right", fontsize=10)
                plt.axhline(y=(np.mean(total_counts) if total_counts else 0.0), ls='dotted', c='gray')
                t = plt.annotate(forplot, xy=(0.02, 0.96), xycoords='axes fraction',
                                 color='gray', fontsize=10, fontweight='bold')
                t.set_bbox(dict(facecolor='whitesmoke', alpha=1.0, edgecolor='whitesmoke',
                                boxstyle="square,pad=0.2"))
                plt.tight_layout()
                plt.savefig(f"statcount_{varName}-{varType}_{forplotname}.png",
                            bbox_inches='tight', dpi=100)
                plt.clf(); plt.close(fig)
            else:
                if chans_sorted:
                    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
                    stack = np.vstack([per_channel_counts[c] for c in chans_sorted]) if chans_sorted else np.zeros((1, len(DayHour)))
                    ax.stackplot(x, stack, labels=[f"ch {c}" for c in chans_sorted], alpha=0.7)
                    ax.plot(x, total_counts, "k-", lw=1.2, label="Total")
                    ax.set_ylabel("Total Observations (" + ("no IQC" if noiqc else "iuse=idqc==0") + ")")
                    ax.set_xlabel("Date (DayHour)")
                    ax.set_xticks(ticks); ax.set_xticklabels([DayHour[i] for i in ticks])
                    ax.set_title(f"{varName}-{varType}  |  {instr}", loc="left", fontsize=10)
                    ax.set_title(f"{datei.strftime('%d%b')}-{datef.strftime('%d%b %Y')}", loc="right", fontsize=10)
                    ax.legend(loc="upper left", ncol=min(4, len(chans_sorted)+1), fontsize=8)
                    t = plt.annotate(forplot, xy=(0.02, 0.96), xycoords='axes fraction',
                                     color='gray', fontsize=10, fontweight='bold')
                    t.set_bbox(dict(facecolor='whitesmoke', alpha=1.0, edgecolor='whitesmoke',
                                    boxstyle="square,pad=0.2"))
                    plt.tight_layout()
                    plt.savefig(f"statcount_{varName}-{varType}_{forplotname}.png",
                                bbox_inches='tight', dpi=100)
                    plt.clf(); plt.close(fig)
                else:
                    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
                    ax.plot(x, total_counts, "-o")
                    ax.set_ylabel("Total Observations (" + ("no IQC" if noiqc else "iuse=idqc==0") + ")")
                    ax.set_xlabel("Date (DayHour)")
                    ax.set_xticks(ticks); ax.set_xticklabels([DayHour[i] for i in ticks])
                    ax.set_title(f"{varName}-{varType}  |  {instr}", loc="left", fontsize=10)
                    ax.set_title(f"{datei.strftime('%d%b')}-{datef.strftime('%d%b %Y')}", loc="right", fontsize=10)
                    plt.axhline(y=(np.mean(total_counts) if total_counts else 0.0), ls='dotted', c='gray')
                    t = plt.annotate(forplot, xy=(0.02, 0.96), xycoords='axes fraction',
                                     color='gray', fontsize=10, fontweight='bold')
                    t.set_bbox(dict(facecolor='whitesmoke', alpha=1.0, edgecolor='whitesmoke',
                                    boxstyle="square,pad=0.2"))
                    plt.tight_layout()
                    plt.savefig(f"statcount_{varName}-{varType}_{forplotname}.png",
                                bbox_inches='tight', dpi=100)
                    plt.clf(); plt.close(fig)
    
        # Mapa (mantido por compat; opcional)
        if figMap:
            try:
                # conta total por ponto na última data com dados (representativo)
                last_idx = None
                for i in range(len(DayHour)-1, -1, -1):
                    if info_check.get(DayHour[i], False):
                        last_idx = i
                        break
                if last_idx is not None:
                    df_last = self[last_idx].obsInfo[varName].query(base_mask).loc[varType]
                    if _safe_len(df_last) > 0 and {'lat','lon'}.issubset(df_last.columns):
                        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
                        ax.scatter(df_last['lon'], df_last['lat'], s=2, alpha=0.5)
                        ax.set_title(f"{varName}-{varType}  |  {instr}  |  {forplot}")
                        ax.set_xlabel("lon"); ax.set_ylabel("lat")
                        plt.tight_layout()
                        plt.savefig(f"statcount_map_{varName}-{varType}_{forplotname}.png",
                                    bbox_inches='tight', dpi=100)
                        plt.clf(); plt.close(fig)
            except Exception:
                pass
    
        print(" Done!\n")
        return


    
    def statcount_Old(self, varName=None, varType=None, noiqc=False, dateIni=None, dateFin=None, nHour="06", channel=None, figTS=False, figMap=False, area=None, **kwargs):

        '''
        The StatCount function plots a time series of assimilated, monitored and rejected data. 

        Example:

        varName = 'uv'           # Variable
        varType = 224            # Source Type
        noiqc = False            # noiqc GSI namelist parameter (OI QC - True or False)
        dateIni = 2013010100     # Inicial Date
        dateFin = 2013010900     # Final Date
        nHour = "06"             # Time Interval
        channel = None           # Radiance channel number (None for the conventional dataset)
        figTS = True             # Creates the time series plot
        figMap = False           # Creates the spatial plot for each time
        
        ! Case conventional dataset: channel = None
        ! The QC process creates a number indicating the data quality for each observation.
        ! These numbers are called QC markers in PrepBUFR files and are important as parts of
        ! the observation information. GSI uses QC markers to decide how to use the data. A 
        ! brief summary of the meaning of the QC markers is as follows:
        ! 
        !    +-----------------+-----------------------------------------------------------+
        !    | QC markes range | Data Process in GSI                                       |
        !    +-----------------+-----------------------------------------------------------+
        !    |  > 15 or        |GSI skips these observations during reading procedure. That|
        !    |  <= 0           |means these observations are tossed                        | 
        !    +-----------------+-----------------------------------------------------------+
        !    |  >= lim_qm      |These observations will be in monitoring status. That means|
        !    |  and            |these observations will be read in and be processed through|
        !    |  < = 15         |GSI QC process (gross check) and innovation calculation    | 
        !    |                 |stage but will not be used in inner iteration.             |
        !    +-----------------+-----------------------------------------------------------+
        !    |  > 0            |Observations will be used in further gross check (failure  |
        !    |  and            |observation will be list in rejection), innovation         |
        !    |  < lim_qm       |caalculation, and the analysis (inner iteration).          |
        !    +-----------------+-----------------------------------------------------------+
        !    +----------------------+---------------+---------------+
        !    |The value of namelist | lim_qm for Ps | lim_qm others |
        !    |option noiqc          |               |               |
        !    +----------------------+---------------+---------------+
        !    |True (without OI QC)  |       7       |       8       |
        !    +----------------------+---------------+---------------+
        !    |False (with OI QC)    |       4       |       4       |
        !    +----------------------+---------------+---------------+
        
        
        ! Case radiance dataset: channel = number
        ! There are three types of data classification: assimilated, monitored and rejected.
        ! Monitored data is organized into two groups: possibly assimilated and possibly rejected.
        !
        !    +------------------------+-------------+--------------------+
        !    |                        |   idqc      |        iuse        |
        !    +------------------------+-------------+--------------------+
        !    | Assimilated            |   == 0      |   >= 1             |
        !    +------------------------+-------------+--------------------+
        !    |            assimilated |   == 0      |   >= -1 and < 1    |
        !    | Monitored              |             |                    |
        !    |            rejected    |   != 0      |   >= -1 and < 1    |
        !    +------------------------+-------------+--------------------+
        !    | Rejected               |   != 0      |   >= 1             |
        !    +------------------------+-------------+--------------------+
        '''
        
        # bibliotecas para criar mapa com cartopy
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        import geopandas


        if(noiqc):
            lim_qm = 8
            if(varName == 'ps'):
                lim_qm = 7
        else:
            lim_qm = 4

        varInfo = getVarInfo(varType, varName, 'instrument')
        if varInfo is not None:
            instrument_title = str(varName) + '-' + str(varType) + '  |  ' + varInfo
        else:
            instrument_title = str(varName) + '-' + str(varType) + '  |  ' + 'Unknown instrument'

        datei = datetime.strptime(str(dateIni), "%Y%m%d%H")
        datef = datetime.strptime(str(dateFin), "%Y%m%d%H")
        date  = datei

        assi, reje, moni, DayHour_tmp = [], [], [], []
        moniAssi, moniReje = [], []
        assif, rejef, monif, moniAssif, moniRejef = [], [], [], [], []
        f = 0
        while (date <= datef):

            datefmt = date.strftime("%Y%m%d%H")
            DayHour_tmp.append(date.strftime("%d%H"))
            
            # try: For issues reading the file (file not found)
            # in the except statement an error message is printed and continues for other dates
            try:
                
                if(channel == None):  # Conventional
                    exp = "(iusev==1)"
                    assim = self[f].obsInfo[varName].loc[varType].query(exp)
                    exp = "(iusev==-1) & (iqc >= "+str(lim_qm)+" and iqc <= 15)"
                    monit = self[f].obsInfo[varName].loc[varType].query(exp)
                    exp = "(iusev==-1) & ((iqc > 15 or iqc <= 0) or (iqc > 0 and iqc < "+str(lim_qm)+"))"
                    rejei = self[f].obsInfo[varName].loc[varType].query(exp)
                
                    assi.append(len(assim))
                    moni.append(len(monit))
                    reje.append(len(rejei))
                    
                    print('')
                    print('assi = ', len(assim), 'moni = ', len(moni), 'reje = ', len(reje))
                
                    if (figMap):
                        df_list = [assim, monit, rejei]
                        name_list = ["Assimilated ["+str(len(assim))+"]","Monitored ["+str(len(monit))+"]","Rejected ["+str(len(rejei))+"]"]
                        marker_list = [".","x","*"]     
                        color_list = ["green","blue","red"]
                    
                        setColor = 0 
                        legend_labels = []
                    
                        #fig = plt.figure(figsize=(12, 6))
                        #ax  = fig.add_subplot(1, 1, 1)
                        #ax = geoMap(area=None,ax=ax)
                        
                        plt.style.use('seaborn-v0_8')
                        fig = plt.figure(figsize=(12, 6))
                        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
                        ax.add_feature(cfeature.COASTLINE)
                        ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor="0.2", zorder=1)
                        ax.add_feature(cfeature.LAND, facecolor="0.95", edgecolor="0.5", linewidth=0.5, zorder=0)
                        ax.coastlines(resolution="110m", linewidth=0.6, zorder=2)
                        
                        print('')
                        print('antes area')
                        
                        if area is not None:
                            ax.set_extent(area, ccrs.PlateCarree())
                        else:
                            ax.set_global()
                            
                        print('')
                        print('após area')
                             
                        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.3, color='gray', alpha=0.5, zorder=3)#, linestyle='--')
                        print('')
                        print('após gridlineas')
                        # versão Read.Radi
                        gl.ylabels_right = False
                        gl.xlabels_top = False
                        # Versão Gerd
                        gl.top_labels = False
                        gl.right_labels = False
                        
                        ax.text(-0.05, 0.55, 'latitude', va='bottom', ha='center', rotation='vertical', rotation_mode='anchor', 
                                  transform=ax.transAxes)
                        ax.text(0.5, -0.08, 'longitude', va='bottom', ha='center', rotation='horizontal', rotation_mode='anchor', 
                                  transform=ax.transAxes)
                        print('')
                        print('após text lat lon')
                        
                        for dfi,namedf,mk,cl in zip(df_list,name_list,marker_list,color_list):
                            df    = dfi
                            legend_labels.append(mpatches.Patch(color=cl, label=namedf) )
                            ax = df.plot(ax=ax,legend=True, marker=mk, color=cl, **kwargs)
                            setColor += 1
                            plt.legend(handles=legend_labels, numpoints=1, loc='lower center', bbox_to_anchor=(0.5, -0.02), 
                                    fancybox=True, shadow=False, frameon=False, ncol=3, prop={"size": 10})
                    
                        date_title = str(date.strftime("%d%b%Y - %H%M")) + ' GMT'
                        plt.title(date_title, loc='right', fontsize=10)
                        plt.title(instrument_title, loc='left', fontsize=9)
                    
                        plt.tight_layout()
                        plt.savefig('TotalObs_'+str(varName) + '-' + str(varType)+'_'+datefmt+'.png', bbox_inches='tight', dpi=100)
                
                
                else:   # Radiance
                    exp = "(nchan=="+str(channel)+") & (iuse >= 1 & idqc==0.0)"
                    assim = self[f].obsInfo[varName].loc[varType].query(exp)
                    exp = "(nchan=="+str(channel)+") & ((iuse >= -1 and iuse < 1) & idqc==0.0)"
                    monitAssim = self[f].obsInfo[varName].loc[varType].query(exp)
                    exp = "(nchan=="+str(channel)+") & ((iuse >= -1 and iuse < 1) & idqc!=0.0)"
                    monitRejei = self[f].obsInfo[varName].loc[varType].query(exp)
                    exp = "(nchan=="+str(channel)+") & (iuse >= 1 & idqc!=0.0)"
                    rejei = self[f].obsInfo[varName].loc[varType].query(exp)
                
                    assi.append(len(assim))
                    moniAssi.append(len(monitAssim))
                    moniReje.append(len(monitRejei))
                    reje.append(len(rejei))
                    
                    forplot = 'Channel ='+str(channel)
                
                    # Radiance plots
                    if (figMap):
                        # Case: assimilated and rejected
                        if ((len(assim)) != 0 or (len(rejei)) != 0):
                            df_list = [assim, rejei]    
                            name_list = ["Assimilated ["+str(len(assim))+"]","Rejected ["+str(len(rejei))+"]"]
                            marker_list = ["^","v"]    
                            color_list = ["green","red"]
                    
                            setColor = 0 
                            legend_labels = []
                    
                            #fig = plt.figure(figsize=(12, 6))
                            #ax  = fig.add_subplot(1, 1, 1)
                            #ax = geoMap(area=None,ax=ax)
                            
                            plt.style.use('seaborn-v0_8')
                            fig = plt.figure(figsize=(12, 6))
                            ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
                            ax.add_feature(cfeature.COASTLINE)
                            ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor="0.2", zorder=1)
                            ax.add_feature(cfeature.LAND, facecolor="0.95", edgecolor="0.5", linewidth=0.5, zorder=0)
                            ax.coastlines(resolution="110m", linewidth=0.6, zorder=2)

                            print('')
                            print('antes area')

                            if area is not None:
                                ax.set_extent(area, ccrs.PlateCarree())
                            else:
                                ax.set_global()

                            print('')
                            print('após area')

                            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.3, color='gray', alpha=0.5, zorder=3)#, linestyle='--')
                            print('')
                            print('após gridlineas')
                            # versão Read.Radi
                            gl.ylabels_right = False
                            gl.xlabels_top = False
                            # Versão Gerd
                            gl.top_labels = False
                            gl.right_labels = False

                            ax.text(-0.05, 0.55, 'latitude', va='bottom', ha='center', rotation='vertical', rotation_mode='anchor', 
                                      transform=ax.transAxes)
                            ax.text(0.5, -0.08, 'longitude', va='bottom', ha='center', rotation='horizontal', rotation_mode='anchor', 
                                      transform=ax.transAxes)
                            print('')
                            print('após text lat lon')
                            
                            
                            
                            for dfi,namedf,mk,cl in zip(df_list,name_list,marker_list,color_list):
                                df    = dfi
                                legend_labels.append(mpatches.Patch(color=cl, label=namedf) )
                                ax = df.plot(ax=ax,legend=True, marker=mk, color=cl, **kwargs) 
                                setColor += 1
                                plt.legend(handles=legend_labels, numpoints=1, loc='lower center', bbox_to_anchor=(0.5, -0.02), 
                                        fancybox=True, shadow=False, frameon=False, ncol=2, prop={"size": 10})
                        
                            date_title = str(date.strftime("%d%b%Y - %H%M")) + ' GMT'
                            plt.title(date_title, loc='right', fontsize=10)
                            plt.title(instrument_title, loc='left', fontsize=9)
                            plt.annotate(forplot, xy=(0.45, 1.015), xytext=(0, 0), xycoords='axes fraction', textcoords='offset points', 
                                         color='gray', fontweight='bold', fontsize='10', horizontalalignment='left', 
                                         verticalalignment='center')
                    
                            plt.tight_layout()
                            plt.savefig('Assim-Rejei_'+str(varName) + '-' + str(varType)+'_'+ 'CH' + str(channel) + '_' +datefmt+'.png', 
                                        bbox_inches='tight', dpi=100)
                        else:
                            print("channel ",channel," not assimilated or rejected on the date -->",date.strftime("%Y-%m-%d:%H"))
                    
                        # Monitored cases: would be assimilated or rejected 
                        if ((len(monitAssim)) != 0 or (len(monitRejei)) != 0):
                            df_list = [monitAssim, monitRejei]
                            name_list = ["Monitored-Assimilated ["+str(len(monitAssim))+"]","Monitored-Rejected ["+str(len(monitRejei))+"]"]
                            marker_list = ["^","v"]   
                            color_list = ["teal","purple"]
                    
                            setColor = 0 
                            legend_labels = []
                    
                            #fig = plt.figure(figsize=(12, 6))
                            #ax  = fig.add_subplot(1, 1, 1)
                            #ax = geoMap(area=None,ax=ax)
                            
                            
                            plt.style.use('seaborn-v0_8')
                            fig = plt.figure(figsize=(12, 6))
                            ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
                            ax.add_feature(cfeature.COASTLINE)
                            ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor="0.2", zorder=1)
                            ax.add_feature(cfeature.LAND, facecolor="0.95", edgecolor="0.5", linewidth=0.5, zorder=0)
                            ax.coastlines(resolution="110m", linewidth=0.6, zorder=2)

                            print('')
                            print('antes area')

                            if area is not None:
                                ax.set_extent(area, ccrs.PlateCarree())
                            else:
                                ax.set_global()

                            print('')
                            print('após area')

                            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.3, color='gray', alpha=0.5, zorder=3)#, linestyle='--')
                            print('')
                            print('após gridlineas')
                            # versão Read.Radi
                            gl.ylabels_right = False
                            gl.xlabels_top = False
                            # Versão Gerd
                            gl.top_labels = False
                            gl.right_labels = False

                            ax.text(-0.05, 0.55, 'latitude', va='bottom', ha='center', rotation='vertical', rotation_mode='anchor', 
                                      transform=ax.transAxes)
                            ax.text(0.5, -0.08, 'longitude', va='bottom', ha='center', rotation='horizontal', rotation_mode='anchor', 
                                      transform=ax.transAxes)
                            print('')
                            print('após text lat lon')
                            
                            
                            for dfi,namedf,mk,cl in zip(df_list,name_list,marker_list,color_list):
                                df    = dfi
                                legend_labels.append(mpatches.Patch(color=cl, label=namedf) )
                                ax = df.plot(ax=ax,legend=True, marker=mk, color=cl, **kwargs) 
                                setColor += 1
                                plt.legend(handles=legend_labels, numpoints=1, loc='lower center', bbox_to_anchor=(0.5, -0.02), 
                                        fancybox=True, shadow=False, frameon=False, ncol=2, prop={"size": 10})
                        
                            date_title = str(date.strftime("%d%b%Y - %H%M")) + ' GMT'
                            plt.title(date_title, loc='right', fontsize=10)
                            plt.title(instrument_title, loc='left', fontsize=9)
                            plt.annotate(forplot, xy=(0.45, 1.015), xytext=(0, 0), xycoords='axes fraction', textcoords='offset points', 
                                         color='gray', fontweight='bold', fontsize='10', horizontalalignment='left', 
                                         verticalalignment='center')
                    
                            plt.tight_layout()
                            plt.savefig('Monitored_'+str(varName) + '-' + str(varType)+'_'+ 'CH' + str(channel) + '_'+datefmt+'.png', 
                                        bbox_inches='tight', dpi=100)
                        else:
                            print("channel ",channel," not monitored on the date -->",date.strftime("%Y-%m-%d:%H"))
                    
            except:
                print("++++++++++++++++++++++++++ ERROR: file reading --> STATCOUNT ++++++++++++++++++++++++++")
                print(setcolor.WARNING + "    >>> No information on this date (" + str(date.strftime("%Y-%m-%d:%H")) +") <<< " + setcolor.ENDC)
                if(channel == None):
                    assi.append(None)
                    moni.append(None)
                    reje.append(None)
                else:
                    assi.append(None)
                    moniAssi.append(None)
                    moniReje.append(None)
                    reje.append(None)
                    
            f = f + 1
            date = date + timedelta(hours=int(nHour))
            date_finale = date
            


        if (figTS):
            if(channel == None):   # Conventional
                if(len(DayHour_tmp) > 4):
                    DayHour = [hr if (ix % int(len(DayHour_tmp) / 4)) == 0 else '' for ix, hr in enumerate(DayHour_tmp)]
                else:
                    DayHour = DayHour_tmp
                
                x_axis      = np.arange(0, len(DayHour), 1)
                date_title = str(datei.strftime("%d%b")) + '-' + str(date_finale.strftime("%d%b")) + ' ' + str(date_finale.strftime("%Y"))
            
                fig = plt.figure(figsize=(6, 4))
                fig, ax1 = plt.subplots(1, 1)
                plt.style.use('seaborn-v0_8-ticks')

                plt.axhline(y=0.0,ls='solid',c='#d3d3d3')

                ax1.plot(x_axis, assi, "o", label="Assimilated \n["+str(sum(assi))+"]", color='green')
                ax1.plot(x_axis, moni, "o", label="Monitored \n["+str(sum(moni))+"]", color='blue')
                ax1.plot(x_axis, reje, "o", label="Rejected \n["+str(sum(reje))+"]", color='red')
                ax1.legend(fancybox=True, frameon=True, shadow=True, loc="upper center",ncol=3)
                ax1.set_xlabel('Date (DayHour)', fontsize=10)
                plt.title(date_title, loc='right', fontsize=10)
                plt.title(instrument_title, loc='left', fontsize=9)
                
                ax1.set_ylim(np.round(-0.05*np.max([assi,moni,reje])), np.round(1.25*np.max([assi,moni,reje])))
                ax1.set_ylabel('Total Observations', color='black', fontsize=10)
                ax1.tick_params('y', colors='black')
                plt.xticks(x_axis, DayHour)
                major_ticks = [ DayHour.index(dh) for dh in filter(None,DayHour) ]
                ax1.set_xticks(major_ticks)
                plt.axhline(y=np.mean(assi),ls='dotted',c='lightgray')
                plt.axhline(y=np.mean(moni),ls='dotted',c='lightgray')
                plt.axhline(y=np.mean(reje),ls='dotted',c='lightgray')
                plt.tight_layout()
                plt.savefig('time_series_'+str(varName) + '-' + str(varType)+'_TotalObs.png', bbox_inches='tight', dpi=100)
                
            else:   # Radiance
                if(len(DayHour_tmp) > 4):
                    DayHour = [hr if (ix % int(len(DayHour_tmp) / 4)) == 0 else '' for ix, hr in enumerate(DayHour_tmp)]
                else:
                    DayHour = DayHour_tmp
                
                x_axis      = np.arange(0, len(DayHour), 1)
                date_title = str(datei.strftime("%d%b")) + '-' + str(date_finale.strftime("%d%b")) + ' ' + str(date_finale.strftime("%Y"))
            
                fig = plt.figure(figsize=(6, 4))
                fig, ax1 = plt.subplots(1, 1)
                plt.style.use('seaborn-v0_8-ticks')

                plt.axhline(y=0.0,ls='solid',c='#d3d3d3')
                
                # List with value None: is removed to calculate sum, max and min
                # The lists below are only used to define the scale of the axes and the total sum of assi/rejei/monit data
                assif     = [x for x in assi if x != None]
                moniAssif = [x for x in moniAssi if x != None]
                moniRejef = [x for x in moniReje if x != None]
                rejef     = [x for x in reje if x != None]

                ax1.plot(x_axis, assi, "o", label="Assimilated \n["+str(sum(assif))+"]", color='green')
                ax1.plot(x_axis, moniAssi, "o", label="Monitored-Assim \n["+str(sum(moniAssif))+"]", color='teal')
                ax1.plot(x_axis, moniReje, "o", label="Monitored-Rejei \n["+str(sum(moniRejef))+"]", color='purple')
                ax1.plot(x_axis, reje, "o", label="Rejected \n["+str(sum(rejef))+"]", color='red')
                ax1.legend(fancybox=True, frameon=True, shadow=True, loc="best",ncol=1)
                ax1.set_xlabel('Date (DayHour)', fontsize=10)
                plt.title(date_title, loc='right', fontsize=10)
                plt.title(instrument_title, loc='left', fontsize=9)
                plt.annotate(forplot, xy=(0.0, 0.965), xytext=(0, 0), xycoords='axes fraction', textcoords='offset points', 
                             color='lightgray', fontweight='bold', fontsize='12', horizontalalignment='left', verticalalignment='center')
                
                ax1.set_ylim(np.round(-0.05*np.max([assif,moniAssif,moniRejef,rejef])),
                             np.round(1.25*np.max([assif,moniAssif,moniRejef,rejef])))
                ax1.set_ylabel('Total Observations', color='black', fontsize=10)
                ax1.tick_params('y', colors='black')
                plt.xticks(x_axis, DayHour)
                major_ticks = [ DayHour.index(dh) for dh in filter(None,DayHour) ]
                ax1.set_xticks(major_ticks)
                plt.axhline(y=np.mean(assif),ls='dotted',c='lightgray')
                plt.axhline(y=np.mean(moniAssif),ls='dotted',c='lightgray')
                plt.axhline(y=np.mean(moniRejef),ls='dotted',c='lightgray')
                plt.axhline(y=np.mean(rejef),ls='dotted',c='lightgray')
                plt.tight_layout()
                plt.savefig('time_series_'+str(varName) + '-' + str(varType) +'_'+ 'CH' + str(channel) + '_'+'_TotalObs.png',
                            bbox_inches='tight', dpi=100)


                
    
#EOC
#-----------------------------------------------------------------------------#

