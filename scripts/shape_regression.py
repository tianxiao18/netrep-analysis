"""Ridge regression from neural shape matrices to augmentation hyperparameters (LOO only)."""

import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import r2_score
from tqdm import tqdm
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from netrep.dualPCA import DualPCA

def center_scale(X):
    Xc = X - np.mean(X, axis=0, keepdims=True)
    return Xc / np.linalg.norm(Xc)


def procrustes_align_to_reference(Y, X_ref):
    U, _, Vt = np.linalg.svd(Y.T @ X_ref)
    return Y @ U @ Vt


def _single_param_display_name(aug_type: str | None) -> str:
    """Axis/title label for ``k == 1`` when ``param_names`` is omitted."""
    if aug_type is None:
        return "σ"
    key = str(aug_type).strip().lower().replace("-", "_")
    if key in {"gaussian_noise", "sp_noise", "noise"}:
        return "σ"
    if key in {"cutout", "crop"}:
        return "W"
    return "σ"


def _try_load_features_X(cache_path: Path, n_rows: int) -> np.ndarray | None:
    if not cache_path.is_file():
        return None
    X = np.load(cache_path, mmap_mode=None)
    if not isinstance(X, np.ndarray) or X.ndim != 2 or int(X.shape[0]) != int(n_rows):
        return None
    return np.asarray(X, dtype=np.float64)


def _save_features_X(cache_path: Path, X: np.ndarray) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, np.asarray(X, dtype=np.float64))


def _select_ridge_alpha_gc(X, y, *, alphas=None, fallback_alpha=1.0):
    """RidgeCV GCV on subset only (needs ≥3 rows)."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(X) < 3:
        return float(fallback_alpha)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    rcv = RidgeCV(alphas=np.logspace(-6, 2, 45) if alphas is None else alphas, cv=None)
    rcv.fit(X, y)
    return float(rcv.alpha_)


def regress_augment_parameter_from_shapes(
    shapes,
    aug_params,
    *,
    reference_shape=None,
    ridge_alpha=1.0,
    dual_pca_n_components=1000,
    features_cache_path=None,
    force_recompute_features=False,
):
    """LOO ridge; ``ridge_alpha="auto"`` uses nested GCV α each fold (median + ``ridge_alpha_per_fold``)."""
    n_rows = len(shapes)

    ref = None
    if reference_shape is not None:
        n_components = min(dual_pca_n_components, len(reference_shape), reference_shape.shape[1])
        if n_components < reference_shape.shape[1]:
            pca = DualPCA(n_components=n_components)
            reference_shape = pca.fit_transform(np.asarray(reference_shape, dtype=np.float32))
        ref = center_scale(np.asarray(reference_shape, dtype=np.float64))

    def _feature_from_shape(S):
        n_components = min(dual_pca_n_components, len(S), S.shape[1])
        if n_components < S.shape[1]:
            pca = DualPCA(n_components=n_components)
            S = pca.fit_transform(S.astype(np.float32)).astype(np.float64)
        M = center_scale(np.asarray(S, dtype=np.float64))
        if ref is not None:
            M = procrustes_align_to_reference(M, ref)
        return np.ravel(M)

    cache_path = Path(features_cache_path) if features_cache_path else None
    X = None
    if cache_path is not None and not force_recompute_features:
        X = _try_load_features_X(cache_path, n_rows)
    if X is None:
        X = np.stack([_feature_from_shape(S) for S in tqdm(shapes, desc="Processing shapes")], axis=0)
        if cache_path is not None:
            _save_features_X(cache_path, X)
            print(f"[features cache] wrote {cache_path}")
    else:
        print(f"[features cache] hit {cache_path}")

    y = np.asarray(aug_params, dtype=np.float64)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    elif y.ndim > 2:
        y = y.reshape(y.shape[0], -1)

    n = min(X.shape[0], y.shape[0])
    X, y = X[:n], y[:n]
    k = y.shape[1]
    auto = ridge_alpha == "auto"
    pred_loo = np.empty_like(y, dtype=np.float64)
    alph_pf = np.empty(n, dtype=np.float64) if auto else None
    ridge_fixed = Ridge(alpha=float(ridge_alpha)) if not auto else None

    for i in tqdm(range(n), desc="nested LOO+GCV" if auto else "LOO ridge"):
        tr = np.arange(n) != i
        Xt, yt = X[tr], y[tr]
        model = Ridge(alpha=_select_ridge_alpha_gc(Xt, yt)) if auto else ridge_fixed
        if auto:
            alph_pf[i] = model.alpha
        model.fit(Xt, yt)
        pred_loo[i : i + 1] = model.predict(X[i : i + 1])

    alpha_out = float(np.median(alph_pf)) if auto else float(ridge_alpha)
    if auto:
        print(f"[ridge] α median={alpha_out:g} ({alph_pf.min():g}-{alph_pf.max():g})")

    pred_all = pred_loo.ravel() if k == 1 else pred_loo
    out = {
        "loo_r2": float(np.mean(r2_score(y, pred_loo, multioutput="raw_values"))),
        "pred_all": pred_all,
        "ridge_alpha": alpha_out,
    }
    if auto:
        out["ridge_alpha_per_fold"] = alph_pf
    return out


def plot_shape_parity(
    aug_params,
    result,
    *,
    overlay_result=None,
    param_names=None,
    aug_type=None,
    save_path=None,
    fig_h=3.8,
):
    """
    LOO parity plots. Multi-target: ``magma`` colormap + per-panel R².

    Optional ``overlay_result``: second regression dict (same ``aug_params``),
    for **single-target** only. Writes **two** figures: ``{stem}_aligned`` (``result``)
    and ``{stem}_unaligned`` (``overlay_result``), e.g. Procrustes vs no reference.

    For ``k == 1``, if ``param_names`` is omitted, pass ``aug_type`` (e.g.
    ``"gaussian_noise"`` → σ, ``"cutout"`` → W) to set axis/title labels.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    # Publication-friendly text sizes (figure saved at dpi=150)
    label_fs = 11
    title_fs = 11
    tick_fs = 10
    r2_fs = 14

    y = np.asarray(aug_params, dtype=np.float64)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    elif y.ndim > 2:
        y = y.reshape(y.shape[0], -1)

    pred = np.asarray(result["pred_all"], dtype=np.float64)
    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)
    elif pred.ndim > 2:
        pred = pred.reshape(pred.shape[0], -1)

    n = min(y.shape[0], pred.shape[0])
    y, pred = y[:n], pred[:n]
    k = min(y.shape[1], pred.shape[1])
    y, pred = y[:, :k], pred[:, :k]

    if param_names is None:
        names = (
            [_single_param_display_name(aug_type)]
            if k == 1
            else [f"param_{j}" for j in range(k)]
        )
    else:
        names = list(param_names)
        if len(names) < k:
            names.extend(f"param_{j}" for j in range(len(names), k))

    def _one_single_panel(fig_path, yt, pr, pname, panel_title, color, marker):
        lo = float(np.nanmin(yt))
        hi = float(np.nanmax(yt))
        lo = float(min(lo, np.nanmin(pr)))
        hi = float(max(hi, np.nanmax(pr)))
        pad = max(1e-9, (hi - lo) * 0.06)
        fig, ax = plt.subplots(1, 1, figsize=(4.0, fig_h))
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="0.45", lw=0.9, zorder=0)
        ax.scatter(
            yt,
            pr,
            c=color,
            s=46,
            marker=marker,
            zorder=2,
            edgecolors="0.15",
            linewidths=0.4,
        )
        panel_r2 = r2_score(yt, pr)
        r2_txt = rf"$R^2 = {panel_r2:.3f}$" if np.isfinite(panel_r2) else r"$R^2 = \mathrm{N/A}$"
        ax.set_xlabel(f"true {pname}", fontsize=label_fs)
        ax.set_ylabel(f"pred {pname}", fontsize=label_fs)
        ax.set_title(panel_title, fontsize=title_fs)
        ax.text(
            0.03,
            0.97,
            r2_txt,
            transform=ax.transAxes,
            fontsize=r2_fs,
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.72", alpha=0.93),
            zorder=6,
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(top=False, right=False, labelsize=tick_fs)
        ax.set_aspect("auto")
        fig.tight_layout()
        if fig_path is not None:
            fig.savefig(fig_path, bbox_inches="tight", dpi=150)
        plt.close(fig)

    if overlay_result is not None and k == 1:
        pra = np.asarray(overlay_result["pred_all"], dtype=np.float64).ravel()[:n]
        yt = y[:, 0]
        pr_a = pred[:, 0]
        m = min(len(yt), len(pr_a), len(pra))
        yt, pr_a, pra = yt[:m], pr_a[:m], pra[:m]
        p0 = names[0]
        if save_path is not None:
            p = Path(save_path)
            path_aln = p.with_name(f"{p.stem}_aligned{p.suffix}")
            path_unaln = p.with_name(f"{p.stem}_unaligned{p.suffix}")
        else:
            path_aln, path_unaln = None, None
        _one_single_panel(path_aln, yt, pr_a, p0, f"{p0} (aligned)", "#31688e", "o")
        _one_single_panel(path_unaln, yt, pra, p0, f"{p0} (unaligned)", "#35b779", "s")
        return

    fig_w = 4.0 * max(1, k)
    fig, axes = plt.subplots(1, k, figsize=(fig_w, fig_h), squeeze=False)

    for j in range(k):
        ax = axes[0, j]
        yt, pr = y[:, j], pred[:, j]

        lo = float(np.nanmin(yt))
        hi = float(np.nanmax(yt))
        lo = float(min(lo, np.nanmin(pr)))
        hi = float(max(hi, np.nanmax(pr)))
        pad = max(1e-9, (hi - lo) * 0.06)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="0.45", lw=0.9, zorder=0)

        if k >= 2:
            c_other = y[:, (j + 1) % k]
            v0, v1 = float(np.nanmin(c_other)), float(np.nanmax(c_other))
            if v0 == v1:
                v0, v1 = v0 - 1.0, v1 + 1.0
            norm = Normalize(vmin=v0, vmax=v1)
            ax.scatter(
                yt,
                pr,
                c=c_other,
                cmap="viridis",
                norm=norm,
                s=44,
                marker="o",
                edgecolors="0.2",
                linewidths=0.35,
                zorder=2,
            )
            mappable = ScalarMappable(norm=norm, cmap="viridis")
            mappable.set_array([])
            cb = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.06)
            cb.set_label(names[(j + 1) % k], fontsize=label_fs)
            cb.ax.tick_params(labelsize=tick_fs)
            panel_r2 = r2_score(yt, pr)
            r2_txt = rf"$R^2 = {panel_r2:.3f}$" if np.isfinite(panel_r2) else r"$R^2 = \mathrm{N/A}$"
        else:
            ax.scatter(
                yt, pr,
                c="#b85c38", s=44, marker="o", zorder=2, edgecolors="0.2", linewidths=0.35
            )
            panel_r2 = r2_score(yt, pr)
            r2_txt = rf"$R^2 = {panel_r2:.3f}$" if np.isfinite(panel_r2) else r"$R^2 = \mathrm{N/A}$"
        ax.set_xlabel(f"true {names[j]}", fontsize=label_fs)
        ax.set_ylabel(f"pred {names[j]}", fontsize=label_fs)
        ax.set_title(names[j], fontsize=title_fs)
        if r2_txt:
            ax.text(
                0.03,
                0.97,
                r2_txt,
                transform=ax.transAxes,
                fontsize=r2_fs,
                verticalalignment="top",
                horizontalalignment="left",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.72", alpha=0.93),
                zorder=6,
            )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(top=False, right=False, labelsize=tick_fs)
        ax.set_aspect("auto")

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def main():
    layer = "layer2.1"

    base = "/mnt/home/the10/ceph/results/netrep/results_aggregated"
    script_dir = Path(__file__).resolve().parent
    feat_cache_dir = script_dir / "shape_regression_features_cache"
    feat_cache_dir.mkdir(parents=True, exist_ok=True)
    layer_tag = layer.replace(".", "_")

    def loo(dat, yvals, ref, fname):
        return regress_augment_parameter_from_shapes(
            dat,
            yvals,
            ridge_alpha="auto",
            reference_shape=ref,
            dual_pca_n_components=1000,
            features_cache_path=feat_cache_dir / fname,
        )

    def load_large_array(path, chunk_size=1, dtype="float64"):
        arr = np.load(path, mmap_mode="r")
        total_shape = arr.shape[0]
        chunks = []
        for start in tqdm(range(0, total_shape, chunk_size), desc="Loading large array"):
            end = min(start + chunk_size, total_shape)
            chunks.append(np.array(arr[start:end], dtype=dtype))
        return np.concatenate(chunks, axis=0)

    reference_shape = np.load(f"{base}/{layer}.npy", mmap_mode="r")[0]
    gaussian_noise_data = load_large_array(f"{base}/gaussian_noise_{layer}.npy")
    cutout_data = load_large_array(f"{base}/cutout_{layer}.npy")
    # patch_gaussian_data = load_large_array(f"{base}/patch_gaussian_{layer}.npy")
    
    print(f"gaussian_noise_data: {gaussian_noise_data.shape}")
    print(f"cutout_data: {cutout_data.shape}")
    # print(f"patch_gaussian_data: {patch_gaussian_data.shape}")

    # reference_shape = individual_data[0]
    aug_param_values = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2]
    saved = loo(gaussian_noise_data, aug_param_values, reference_shape, f"gaussian_noise_{layer_tag}_ref_X.npy")
    saved_no_ref = loo(gaussian_noise_data, aug_param_values, None, f"gaussian_noise_{layer_tag}_noref_X.npy")
    print(
        f"gaussian_noise R² {saved['loo_r2']:.4f} / {saved_no_ref['loo_r2']:.4f}, "
        f"α {saved['ridge_alpha']:.4g} / {saved_no_ref['ridge_alpha']:.4g}"
    )
    plot_shape_parity(
        aug_param_values,
        saved,
        overlay_result=saved_no_ref,
        aug_type="gaussian_noise",
        save_path=script_dir / "shape_regression_gaussian_noise.png",
    )

    aug_param_values = [2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 20.0, 24.0]
    saved = loo(cutout_data, aug_param_values, reference_shape, f"cutout_{layer_tag}_ref_X.npy")
    saved_no_ref = loo(cutout_data, aug_param_values, None, f"cutout_{layer_tag}_noref_X.npy")
    print(f"cutout R² {saved['loo_r2']:.4f} / {saved_no_ref['loo_r2']:.4f}")
    plot_shape_parity(
        aug_param_values,
        saved,
        overlay_result=saved_no_ref,
        aug_type="cutout",
        save_path=script_dir / "shape_regression_cutout.png",
    )

    patch_gaussian_hypers = np.array(
        [
            [x, y]
            for x in [4.0, 8.0, 10.0, 12.0, 16.0, 20.0, 24.0]
            for y in [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
        ]
    )
    pg_path = f"{base}/patch_gaussian_{layer}.npy"
    if os.path.isfile(pg_path):
        patch_gaussian_shapes = load_large_array(pg_path)[: len(patch_gaussian_hypers)]
        out = loo(patch_gaussian_shapes, patch_gaussian_hypers, reference_shape, f"patch_gaussian_{layer_tag}_ref_X.npy")
        print(out["loo_r2"], out["pred_all"])
        plot_shape_parity(
            patch_gaussian_hypers,
            out,
            param_names=("patch_gaussian_size", "sigma"),
            save_path=script_dir / "shape_regression_patch_gaussian.png",
        )
        out_no_ref = loo(
            patch_gaussian_shapes, patch_gaussian_hypers, None, f"patch_gaussian_{layer_tag}_no_ref_X.npy"
        )
        print(out_no_ref["loo_r2"], out_no_ref["pred_all"])
        plot_shape_parity(
            patch_gaussian_hypers,
            out_no_ref,
            param_names=("patch_gaussian_size", "sigma"),
            save_path=script_dir / "shape_regression_patch_gaussian_no_ref.png",
        )
    else:
        print(f"[skip] patch gaussian: not found {pg_path}")


if __name__ == "__main__":
    main()
