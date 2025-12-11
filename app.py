import io, os, zipfile, warnings, csv
import math
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.set_page_config(page_title="Yoshihara Lab SOP Software YH#2 qPCR_Analysis_v0.2.3_QuantStudio6", layout="wide")
st.title("🧬 Yoshihara Lab SOP Software YH#2 qPCR_Analysis_v0.2.3_QuantStudio6")

# ---------- Core helpers ----------
def read_qpcr_textfile(content_bytes: bytes) -> pd.DataFrame:
    # --- Try UTF-8 or CP932 ---
    text = None
    for enc in ("utf-8", "cp932"):
        try:
            text = content_bytes.decode(enc)
            break
        except Exception:
            pass
    if text is None:
        raise ValueError("Encoding error (utf-8/cp932 Failed)")

    lines = text.splitlines()

    # --- Find header line (行のどこかに Well が含まれていればOK) ---
    header_idx = None
    for i, l in enumerate(lines):
        if "Well" in l.split():
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("Header line containing 'Well' not found")

    header_line = lines[header_idx]

    # --- 区切り文字を推定 ---
    if "," in header_line:
        sep = ","
    elif "\t" in header_line:
        sep = "\t"
    else:
        sep = r"\s+"

    df = pd.read_csv(io.StringIO("\n".join(lines[header_idx:])), sep=sep, engine="python")
    df.columns = [c.strip() for c in df.columns]

    # Cq 列名のゆらぎに対応
    if "Cq" not in df.columns:
        cq_col = next((c for c in df.columns if c.lower() in ("cq", "ct", "cq mean")), None)
        if cq_col:
            df.rename(columns={cq_col: "Cq"}, inplace=True)
        else:
            raise ValueError("Cq column not found")

    # Target, Task も念のためそろえておく
    if "Target" not in df.columns:
        tgt = next((c for c in df.columns if "target" in c.lower() or "detector" in c.lower()), None)
        if tgt:
            df.rename(columns={tgt: "Target"}, inplace=True)

    if "Task" not in df.columns:
        task = next((c for c in df.columns if c.lower() == "task"), None)
        if task:
            df.rename(columns={task: "Task"}, inplace=True)

    return df

def clean_dataframe_for_analysis(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy(); df.columns = [c.strip() for c in df.columns]
    def _rn(target, cond):
        if target in df.columns: return
        cands = [c for c in df.columns if cond(c)]
        if cands: df.rename(columns={cands[0]:target}, inplace=True)
    _rn("Reporter", lambda c:c.lower()=="reporter")
    _rn("Cq", lambda c:c.lower() in ("ct", "cq"))
    _rn("Sample", lambda c:c.lower() in ("sample","sample_name","samplename"))
    _rn("Target", lambda c:"detector" in c.lower() or c.lower()=="target")
    _rn("Task", lambda c:c.lower()=="task")
    _rn("Quantity", lambda c:"quantity" in c.lower())

    # Undetermined → NaN
    df["Cq"] = pd.to_numeric(df["Cq"].replace({"Undetermined":np.nan,"undetermined":np.nan}), errors="coerce")
    df["Quantity"] = pd.to_numeric(df.get("Quantity", np.nan), errors="coerce")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if "Well" in df.columns:
            df["Well"] = pd.to_numeric(df["Well"], errors="ignore")
    return df

def compute_standard_curve(df_std: pd.DataFrame):
    d = df_std.replace([np.inf, -np.inf], np.nan).dropna(subset=["Cq","Quantity"]).copy()
    d = d[d["Quantity"] > 0]
    if len(d) < 2:
        return None
    X = np.log10(d["Quantity"].values).reshape(-1,1)
    y = d["Cq"].values
    model = LinearRegression().fit(X,y)
    r2 = r2_score(y, model.predict(X))
    return {"slope": float(model.coef_[0]), "intercept": float(model.intercept_), "r2": float(r2), "model": model}

def cq_to_quantity(cq, slope, intercept):
    if pd.isna(cq) or slope == 0 or np.isnan(slope):
        return np.nan
    return float(10 ** ((cq - intercept) / slope))

# ---------- State ----------
if "df_raw" not in st.session_state:
    st.session_state.update({
        "df_raw": None,
        "df_std_clean": None,
        "df_smp": None,
        "df_smp_updated": None,
        "conditions": ["Control","Treatment1","Treatment2"]
    })

# ---------- UI: Step tabs ----------
t1, t2, t3, t4, t5, t6 = st.tabs(["1) Upload", "2) Clean Standards", "3) Curves",
                                  "4) Assign", "5) Quantify", "6) Export"])

# 1) Upload —— 複数ファイル対応版（安定化）
with t1:
    ups = st.file_uploader(
        "📄 Upload qPCR result files (multiple allowed / TXT・TSV・CSV）",
        type=["txt", "tsv", "csv"],
        accept_multiple_files=True,
        key="uploader_multi",
        help="You can combine multiple exports from the same run."
    )

    col_l, col_r = st.columns([1,1])
    load_clicked = col_l.button("Load file(s)")
    clear_clicked = col_r.button("Clear")

    if clear_clicked:
        st.session_state.df_raw = None
        st.experimental_rerun()

    if load_clicked:
        if not ups:
            st.warning("No files selected.")
        else:
            df_list, errs = [], []
            for up in ups:
                try:
                    content_bytes = up.getvalue()  # robust vs .read()
                    df_tmp = read_qpcr_textfile(content_bytes)
                    df_tmp = clean_dataframe_for_analysis(df_tmp)
                    df_tmp["SourceFile"] = up.name
                    df_list.append(df_tmp)
                except Exception as e:
                    errs.append(f"{up.name}: {e}")

            if errs:
                st.error("Some files failed to load. Details below:")
                st.code("\n".join(errs))

            if df_list:
                df = pd.concat(df_list, axis=0, ignore_index=True)
                df = df.dropna(how="all").reset_index(drop=True)
                st.session_state.df_raw = df
                st.success(f"Loaded {len(df_list)} file(s). Total rows = {len(df):,}")
            else:
                st.warning("No readable files were found.")

    # プレビュー & 必須列チェック
    if st.session_state.get("df_raw") is not None:
        st.caption("Preview of the first rows (max 30)")
        st.dataframe(st.session_state.df_raw.head(30), use_container_width=True)

        need = {"Task", "Cq", "Target"}
        miss = [c for c in need if c not in st.session_state.df_raw.columns]
        if miss:
            st.error(f"Missing required columns: {miss}")
        else:
            cols = [c for c in ["Target", "Task", "SourceFile"] if c in st.session_state.df_raw.columns]
            if cols:
                with st.expander("Summary（Detector/Task/SourceFile）", expanded=False):
                    st.write(
                        st.session_state.df_raw[cols]
                        .value_counts()
                        .reset_index(name="count")
                    )

# 2) Clean Standards
with t2:
    if st.session_state.df_raw is None:
        st.info("Please Complete Upload")
    else:
        df_std = st.session_state.df_raw.copy()
        df_std = df_std[df_std["Task"].astype(str).str.lower()=="standard"].dropna(subset=["Cq"]).copy()
        if "Well" in df_std.columns: df_std["Well"] = pd.to_numeric(df_std["Well"], errors="coerce")
        df_std = df_std.sort_values(["Target","Quantity","Well"], na_position="last")
        drops = []
        for (det, qty), sub in df_std.groupby(["Target","Quantity"], dropna=False):
            st.markdown(f"**{det} — Qty {qty}**  (ΔCq={sub['Cq'].max()-sub['Cq'].min():.2f})")
            show = sub[["Well","Sample","Cq"]].reset_index()
            idxs = st.multiselect("Drop rows", options=show["index"].tolist(),
                                  format_func=lambda i: f"Well {int(df_std.loc[i,'Well']) if pd.notna(df_std.loc[i,'Well']) else '?'} / {df_std.loc[i,'Sample']} (Cq={df_std.loc[i,'Ct']})",
                                  key=f"drop_{det}_{qty}")
            drops += idxs
            st.dataframe(show.drop(columns=["index"]), use_container_width=True)
        if st.button("Apply cleaning"):
            clean = df_std.drop(index=drops).reset_index(drop=True)
            st.session_state.df_std_clean = clean
            st.success(f"Cleaned: {len(df_std)} → {len(clean)} rows")
        if st.session_state.df_std_clean is not None:
            st.dataframe(st.session_state.df_std_clean.head(20), use_container_width=True)

# 3) Standard Curves
with t3:
    if st.session_state.df_std_clean is None:
        st.info("2) Please do Clean Standards")
    else:
        buf_pdf = io.BytesIO()
        with PdfPages(buf_pdf) as pdf:
            for det in st.session_state.df_std_clean["Target"].dropna().unique():
                ddf = st.session_state.df_std_clean[st.session_state.df_std_clean["Target"]==det]
                sc = compute_standard_curve(ddf)
                fig, ax = plt.subplots(figsize=(5,3.2))
                ax.set_title(f"STANDARD curve: {det}")
                ax.set_xlabel("log10(Quantity)"); ax.set_ylabel("Cq")
                if sc:
                    x = np.log10(ddf["Quantity"]); y = ddf["Cq"]
                    ax.scatter(x,y, color="black", s=12)
                    xx = np.linspace(x.min(), x.max(), 100).reshape(-1,1)
                    ax.plot(xx, sc["model"].predict(xx), "--", linewidth=0.8, color="black")
                    ax.text(0.02,0.02,f"slope={sc['slope']:.3f}\nR²={sc['r2']:.3f}", transform=ax.transAxes)
                else:
                    ax.text(0.5,0.5,"Insufficient points", ha="center")
                st.pyplot(fig, clear_figure=True); pdf.savefig(fig); plt.close(fig)
        st.download_button("📄 Download standard-curve report (PDF)",
                           data=buf_pdf.getvalue(), file_name="qpcr_standard_curve_report.pdf",
                           mime="application/pdf")

# 4) Assign
with t4:
    if st.session_state.df_raw is None:
        st.info("Please Complete Upload")
    else:
        # ---- Build working table: Unknown only ----
        work = st.session_state.df_raw.copy()
        work = work[work["Task"].astype(str).str.lower() == "unknown"].copy()

        # Sort: Detector -> Well (numeric if possible)
        if "Well" in work.columns:
            work["Well"] = pd.to_numeric(work["Well"], errors="coerce")
            work = work.sort_values(["Target", "Well"], na_position="last").reset_index(drop=False)
            base_index = work["index"].to_list()  # original df_raw indices
        else:
            work = work.sort_values(["Target"]).reset_index(drop=False)
            base_index = work["index"].to_list()

        # ---- Session assignment table (Condition/Replicate) ----
        if "assign_df" not in st.session_state:
            st.session_state.assign_df = pd.DataFrame(index=st.session_state.df_raw.index)
            st.session_state.assign_df["Condition"] = ""
            st.session_state.assign_df["Replicate"] = ""

        # Detector-level template store
        if "detector_template" not in st.session_state:
            st.session_state.detector_template = None

        # ---- Conditions editor ----
        st.subheader("Define conditions")
        cond_text = st.text_area(
            "One condition per line",
            value="\n".join(st.session_state.get("conditions", ["Control", "Treatment1", "Treatment2"])),
            height=100,
            help="Editing here updates the select boxes below."
        )
        st.session_state.conditions = [c.strip() for c in cond_text.splitlines() if c.strip()]

        st.markdown("---")

        # ---- Per-detector assignment UI ----
        for det in sorted(work["Target"].dropna().unique().tolist()):
            with st.expander(f"Detector: {det}", expanded=False):
                sub = work[work["Target"] == det].copy().reset_index(drop=True)  # has 'index' (df_raw row id)

                # ====== Filter (Sample/Well/Cq) ======
                view = sub.copy()

                # ====== Multi-select target rows ======
                st.caption("Select samples to apply")
                # display label + original df_raw index as value
                options = []
                for _, r in view.iterrows():
                    well_val = r.get("Well")
                    if pd.isna(well_val):
                        well_str = ""
                    else:
                        try:
                            w = float(well_val)
                            well_str = str(int(w)) if w.is_integer() else str(well_val)
                        except Exception:
                            well_str = str(well_val)
                    label = f"{r['Sample']} (Well={well_str}, Cq={r.get('Cq')})"
                    options.append((label, int(r["index"])))

                sel_key = f"ms_{det}"
                selected = st.multiselect(
                    "Samples",
                    [v for (_, v) in options],
                    format_func=lambda idx: next(lbl for (lbl, v) in options if v == idx),
                    key=sel_key
                )

                # ====== Clean section bar (quick clear tools) ======
                st.markdown("**Clean section**")
                cclean1, cclean2 = st.columns([1, 1])
                with cclean1:
                    if st.button("Clear assignment for SELECTED", key=f"clear_sel_{det}"):
                        if not selected:
                            st.warning("No sample selected.")
                        else:
                            st.session_state.assign_df.loc[selected, ["Condition", "Replicate"]] = ["", ""]
                            st.success(f"Cleared assignments for {len(selected)} row(s).")
                with cclean2:
                    if st.button("Clear assignment for ALL in this detector", key=f"clear_all_{det}"):
                        locs = sub["index"].to_list()
                        st.session_state.assign_df.loc[locs, ["Condition", "Replicate"]] = ["", ""]
                        st.success(f"Cleared assignments for ALL {len(locs)} row(s) in '{det}'.")

                # ====== Clear selection only ======
                if st.button("Clear selection", key=f"selclr_{det}"):
                    st.session_state[sel_key] = []
                    st.experimental_rerun()

                # ====== Apply Condition / Replicate ======
                c1, c2, c3, c4 = st.columns([2, 2, 2, 2])
                with c1:
                    pick_cond = st.selectbox("Condition", st.session_state.conditions, key=f"cond_{det}")
                with c2:
                    pick_rep = st.selectbox("Bio Rep", ["Rep1", "Rep2", "Rep3"], key=f"rep_{det}")
                with c3:
                    apply_clicked = st.button("Apply to selected", key=f"apply_{det}")
                with c4:
                    apply_all_clicked = st.button("Apply to ALL in this detector", key=f"applyall_{det}")

                if apply_clicked:
                    if not selected:
                        st.warning("No sample selected.")
                    else:
                        st.session_state.assign_df.loc[selected, ["Condition", "Replicate"]] = [pick_cond, pick_rep]
                        st.success(f"Applied to {len(selected)} row(s).")

                if apply_all_clicked:
                    locs = sub["index"].to_list()
                    st.session_state.assign_df.loc[locs, ["Condition", "Replicate"]] = [pick_cond, pick_rep]
                    st.success(f"Applied to ALL {len(locs)} row(s) in '{det}'.")

                # ====== Template (copy/paste per detector, size must match) ======
                t1, t2 = st.columns([1, 1])
                if t1.button("Copy as template", key=f"copy_{det}"):
                    templ = st.session_state.assign_df.loc[sub["index"], ["Condition", "Replicate"]].copy().reset_index(drop=True)
                    st.session_state.detector_template = templ
                    st.success(f"Template copied from '{det}' ({len(templ)} rows).")

                if t2.button("Use the template", key=f"paste_{det}"):
                    templ = st.session_state.detector_template
                    tgt = st.session_state.assign_df.loc[sub["index"], ["Condition", "Replicate"]].copy().reset_index(drop=True)
                    if templ is None:
                        st.warning("No template copied yet.")
                    elif len(templ) != len(tgt):
                        st.warning(f"Template size mismatch: template={len(templ)} vs target={len(tgt)}")
                    else:
                        st.session_state.assign_df.loc[sub["index"], ["Condition", "Replicate"]] = templ.values
                        st.success(f"Template applied to '{det}'.")

                # ====== Preview (with current assignments) ======
                st.caption("Updated table (after operations)")
                sub_preview = work[work["Target"] == det][["index", "Sample", "Cq"]].copy()
                sub_preview["Condition"] = st.session_state.assign_df.loc[sub_preview["index"], "Condition"].values
                sub_preview["Replicate"] = st.session_state.assign_df.loc[sub_preview["index"], "Replicate"].values
                sub_preview = sub_preview.drop(columns=["index"])
                st.dataframe(sub_preview, use_container_width=True, hide_index=True)

        st.markdown("---")

        # ---- Save to session (validation) ----
        if st.button("Save assignment", type="primary", key="save_assign"):
            assigned = st.session_state.assign_df.copy()
            target_idx = st.session_state.df_raw.index[
                st.session_state.df_raw["Task"].astype(str).str.lower() == "unknown"
            ]
            missing_mask = (assigned.loc[target_idx, "Condition"] == "") | (assigned.loc[target_idx, "Replicate"] == "")
            if missing_mask.any():
                miss_count = int(missing_mask.sum())
                st.error(f"{miss_count} samples missing assignment (Condition or Replicate).")
            else:
                if "Condition" not in st.session_state.df_raw.columns:
                    st.session_state.df_raw["Condition"] = ""
                if "Replicate" not in st.session_state.df_raw.columns:
                    st.session_state.df_raw["Replicate"] = ""

                st.session_state.df_raw.loc[:, "Condition"] = assigned["Condition"].fillna(st.session_state.df_raw["Condition"])
                st.session_state.df_raw.loc[:, "Replicate"] = assigned["Replicate"].fillna(st.session_state.df_raw["Replicate"])

                st.session_state.df_smp = st.session_state.df_raw[
                    st.session_state.df_raw["Task"].astype(str).str.lower() == "unknown"
                ].copy()

                st.success("Assignments saved.")

# 5) Quantify（helpers を使用：Undetermined/欠損は NaN 扱い）
with t5:
    if st.session_state.df_smp is None or st.session_state.df_std_clean is None:
        st.info("Please Complete 2) & 4)")
    else:
        df_smp = st.session_state.df_smp.copy()
        df_smp["Quantity"] = np.nan

        # --- 1) Cq -> Quantity for each detector via standard curve ---
        for det in st.session_state.df_std_clean["Target"].dropna().unique():
            dstd = st.session_state.df_std_clean[
                (st.session_state.df_std_clean["Target"] == det) &
                (st.session_state.df_std_clean["Task"].astype(str).str.lower() == "standard")
            ].copy()

            sc = compute_standard_curve(dstd)
            if sc is None:
                st.warning(f"Detector '{det}': Not enough standard points or Quantity <= 0. Skipped.")
                continue

            slope, intercept = sc["slope"], sc["intercept"]
            mask = (df_smp["Target"] == det)
            df_smp.loc[mask, "Quantity"] = df_smp.loc[mask, "Cq"].apply(
                lambda c: cq_to_quantity(c, slope, intercept)
            )
            # guard against negative due to numeric glitches
            df_smp.loc[mask & (df_smp["Quantity"] < 0), "Quantity"] = np.nan

        detectors_for_ctrl = sorted(df_smp["Target"].dropna().unique().tolist())
        if not detectors_for_ctrl:
            st.error("Target was not found. Please check Upload/Assign")
        else:
            ctrl_det = st.selectbox("Control detector", detectors_for_ctrl, key="ctrl_det_select")

            if st.button("Run Relative Quantification"):
                # --- 2) Build control table (attach orig_index for order alignment) ---
                ctrl_df = (
                    df_smp[df_smp["Target"] == ctrl_det][["Condition", "Replicate", "Quantity"]]
                    .rename(columns={"Quantity": "Ctrl_Quantity"})
                    .copy()
                )
                # 0 or missing control quantities are invalid denominators
                ctrl_df.loc[(ctrl_df["Ctrl_Quantity"] <= 0) | (ctrl_df["Ctrl_Quantity"].isna()), "Ctrl_Quantity"] = np.nan
                ctrl_df = ctrl_df.sort_index().reset_index().rename(columns={"index": "orig_index"})

                # Fallback: condition-level mean of control
                ctrl_cond_mean = (
                    ctrl_df.groupby("Condition", as_index=False)["Ctrl_Quantity"]
                    .mean()
                    .rename(columns={"Ctrl_Quantity": "Ctrl_Cond_Mean"})
                )

                df_temp = df_smp.copy()
                if "Relative Quantity" not in df_temp.columns:
                    df_temp["Relative Quantity"] = np.nan

                # --- 3) Per-detector relative quantification ---
                for det in df_temp["Target"].dropna().unique():
                    m = (df_temp["Target"] == det)
                    ddet = (
                        df_temp.loc[m, ["Condition", "Replicate", "Quantity"]]
                        .reset_index()
                        .rename(columns={"index": "orig_index"})
                        .sort_values("orig_index").reset_index(drop=True)
                    )

                    # pair with control rows that share Condition + Replicate
                    cond_rep = ddet[["Condition", "Replicate"]].drop_duplicates()
                    ctrl_sub = ctrl_df.merge(cond_rep, on=["Condition", "Replicate"], how="inner")

                    # if we can align one-to-one, keep that order using control's orig_index
                    if len(ctrl_sub) == len(ddet):
                        ctrl_sub = ctrl_sub.sort_values("orig_index").reset_index(drop=True)
                        merged = ddet.copy()
                        merged["Ctrl_Quantity"] = ctrl_sub["Ctrl_Quantity"].values
                        merged["Used_Ctrl"] = merged["Ctrl_Quantity"]
                    else:
                        warnings.warn(
                            f"Detector '{det}': Control and target replicate counts differ "
                            f"({len(ctrl_sub)} vs {len(ddet)}). Using condition-level control mean."
                        )
                        # fallback: merge control (may bring NaN) then fill by condition mean
                        merged = ddet.merge(ctrl_df[["Condition", "Replicate", "Ctrl_Quantity"]],
                                            on=["Condition", "Replicate"], how="left")
                        merged = merged.merge(ctrl_cond_mean, on="Condition", how="left")
                        merged["Used_Ctrl"] = merged["Ctrl_Quantity"].fillna(merged["Ctrl_Cond_Mean"])

                    # invalid denominator or numerator -> NaN
                    invalid_den = merged["Used_Ctrl"].isna() | (merged["Used_Ctrl"] <= 0)
                    invalid_num = merged["Quantity"].isna() | (merged["Quantity"] < 0)
                    merged["Relative Quantity"] = np.where(
                        invalid_den | invalid_num,
                        np.nan,
                        merged["Quantity"] / merged["Used_Ctrl"]
                    )

                    # write back
                    df_temp.loc[merged["orig_index"], "Relative Quantity"] = merged["Relative Quantity"].values

                # --- 4) Mean/SEM per (Detector, Condition, Replicate) ---
                stats = (
                    df_temp.groupby(["Target", "Condition", "Replicate"], observed=False)["Relative Quantity"]
                    .agg(["mean", "sem"]).reset_index()
                    .rename(columns={"mean": "RelQ_Mean", "sem": "RelQ_SEM"})
                )

                st.session_state.df_smp_updated = df_temp.merge(
                    stats, on=["Target", "Condition", "Replicate"], how="left"
                )
                st.success("Relative quantification done.")

        if st.session_state.get("df_smp_updated") is not None:
            st.dataframe(st.session_state.df_smp_updated.head(30), use_container_width=True)

# 6) Export（PDF: 2in×2in グリッド & StandardカーブPDFも同梱 / UIにもプレビュー）
with t6:
    if st.session_state.df_smp_updated is None:
        st.info("Please proceed with 5)")
    else:
        import io, zipfile
        from matplotlib.backends.backend_pdf import PdfPages

        # ---- date-stamped base name ----
        today = pd.Timestamp.now().strftime("%Y-%m-%d")
        base_name = f"{today}qPCR_Results"

        # ---- helper: draw panels (ALWAYS show all conditions; 0 if missing) ----
        def draw_relq_panel(ax, ddf, conds):
            """
            Bar = Mean of 'Relative Quantity' per Condition,
            Error bar = SEM across biological replicates (Replicate) within each Condition.
            Even when a condition has no data, draw a bar at 0 (so it never disappears).
            Replicate dots are plotted only for conditions that have data.
            """
            # 全条件を 0 初期化（データ無しでも棒を描く）
            cond_stats = pd.DataFrame({"Condition": conds, "Mean": 0.0, "SEM": 0.0})

            vals = ddf[["Condition", "Replicate", "Relative Quantity"]].copy()
            usable = vals.dropna(subset=["Relative Quantity"])

            if not usable.empty:
                # 棒グラフ (平均値 ± SEM) を Condition ごとにまとめる）
                rep_means = (
                    usable.groupby(["Condition", "Replicate"], observed=False)["Relative Quantity"]
                          .mean()
                          .reset_index(name="Rep_Mean")
                )
                # Conditionごとの Mean と SEM（= sd/sqrt(n)）
                _cs = (
                    rep_means.groupby("Condition", observed=False)["Rep_Mean"]
                             .agg(Mean="mean", SEM=lambda s: s.std(ddof=1) / np.sqrt(s.count()))
                             .reset_index()
                )

                if not _cs.empty:
                    merged = cond_stats.merge(_cs, on="Condition", how="left", suffixes=("_base", ""))
                    mean_new = merged["Mean"].where(merged["Mean"].notna(), merged["Mean_base"])
                    sem_new  = merged["SEM"].where(merged["SEM"].notna(), merged["SEM_base"])
                    cond_stats = pd.DataFrame({
                        "Condition": merged["Condition"],
                        "Mean": mean_new.fillna(0.0),
                        "SEM":  sem_new.fillna(0.0),
                    })
                else:
                    rep_means = pd.DataFrame(columns=["Condition", "Replicate", "Rep_Mean"])
            else:
                rep_means = pd.DataFrame(columns=["Condition", "Replicate", "Rep_Mean"])

            # 棒（0 も描画）
            ax.bar(
                cond_stats["Condition"], cond_stats["Mean"],
                yerr=cond_stats["SEM"], capsize=2, alpha=0.65, linewidth=0.4,
                error_kw={"elinewidth": 0.25, "capthick": 0.25}
            )

            # 黒ドット（データがある条件のみ）
            if not rep_means.empty:
                cond_to_x = {c: i for i, c in enumerate(cond_stats["Condition"])}
                rep_offset = {"Rep1": -0.12, "Rep2": 0.0, "Rep3": 0.12}
                for _, row in rep_means.iterrows():
                    x = cond_to_x.get(row["Condition"])
                    if x is None:
                        continue
                    ax.scatter(x + rep_offset.get(str(row["Replicate"]), 0.0),
                               row["Rep_Mean"], s=18, zorder=3, color="black")

            ax.set_ylabel("Relative Quantity", fontsize=8)
            ax.set_ylim(bottom=0)
            ax.tick_params(axis="x", labelrotation=45, labelsize=7)
            ax.tick_params(axis="y", labelsize=7)
            for spine in ax.spines.values():
                spine.set_linewidth(0.4)
            return True

        # ---- layout (auto adjust by Condition count) ----
        condition_list = st.session_state.conditions
        dets = st.session_state.df_smp_updated["Target"].dropna().unique().tolist()
        n_detectors = len(dets)
        n_conditions = len(condition_list)

        # 1ページあたり最大の横列数を条件数に応じて可変化
        if n_conditions <= 4:
            max_cols = 4
        elif n_conditions <= 6:
            max_cols = 3
        else:
            max_cols = 2

        # 列数・行数を決定
        n_cols = min(max_cols, n_detectors) if n_detectors > 0 else 1
        n_rows = math.ceil(n_detectors / n_cols) if n_cols > 0 else 1
        PAGE_CAP = n_cols * n_rows  # 1ページのパネル数

        # 各パネルサイズ（固定）
        PANEL_W, PANEL_H = 2.0, 2.0
        FIG_W, FIG_H = n_cols * PANEL_W, n_rows * PANEL_H

        # A) Relative-quantity grid PDF + UI previews (PNG)
        relq_pdf_buf   = io.BytesIO()
        relq_page_pngs = []
        with PdfPages(relq_pdf_buf) as pdf:
            panel_i = 0
            fig = None; axs = None

            for det in dets:
                if panel_i % PAGE_CAP == 0:
                    # 直前ページを保存
                    if fig is not None:
                        fig.tight_layout(pad=0.8); pdf.savefig(fig)
                        buf_png = io.BytesIO()
                        fig.savefig(buf_png, format="png", dpi=200, bbox_inches="tight")
                        relq_page_pngs.append(buf_png.getvalue())
                        plt.close(fig)
                    # 新規ページ
                    fig, ax_grid = plt.subplots(n_rows, n_cols, figsize=(FIG_W, FIG_H))
                    axs = ax_grid.flatten() if hasattr(ax_grid, "flatten") else [ax_grid]

                ax  = axs[panel_i % PAGE_CAP]
                ddf = st.session_state.df_smp_updated[
                    st.session_state.df_smp_updated["Target"] == det
                ].copy()

                draw_relq_panel(ax, ddf, condition_list)
                ax.set_title(det, fontsize=9)
                panel_i += 1

            if fig is not None:
                used = panel_i % PAGE_CAP or PAGE_CAP
                if used < PAGE_CAP:
                    for k in range(used, PAGE_CAP):
                        axs[k].axis("off")
                fig.tight_layout(pad=0.8); pdf.savefig(fig)
                buf_png = io.BytesIO()
                fig.savefig(buf_png, format="png", dpi=200, bbox_inches="tight")
                relq_page_pngs.append(buf_png.getvalue())
                plt.close(fig)

        # B) Standard-curve PDF + previews
        std_pdf_buf   = io.BytesIO()
        std_page_pngs = []
        with PdfPages(std_pdf_buf) as pdf:
            if st.session_state.df_std_clean is not None and not st.session_state.df_std_clean.empty:
                panel_i = 0
                fig = None; axs = None
                for det in st.session_state.df_std_clean["Target"].dropna().unique():
                    ddf   = st.session_state.df_std_clean[
                        st.session_state.df_std_clean["Target"] == det
                    ].copy()
                    dwork = ddf.replace([np.inf, -np.inf], np.nan).dropna(subset=["Cq", "Quantity"])
                    dwork = dwork[dwork["Quantity"] > 0]
                    if len(dwork) < 2:
                        continue

                    X = np.log10(dwork["Quantity"].to_numpy()).reshape(-1, 1)
                    y = dwork["Cq"].to_numpy()
                    model = LinearRegression().fit(X, y)
                    slope = float(model.coef_[0]); intercept = float(model.intercept_)
                    r2    = r2_score(y, model.predict(X))

                    if panel_i % PAGE_CAP == 0:
                        if fig is not None:
                            fig.tight_layout(pad=0.8); pdf.savefig(fig)
                            buf_png = io.BytesIO()
                            fig.savefig(buf_png, format="png", dpi=200, bbox_inches="tight")
                            std_page_pngs.append(buf_png.getvalue())
                            plt.close(fig)
                        fig, ax_grid = plt.subplots(n_rows, n_cols, figsize=(FIG_W, FIG_H))
                        axs = ax_grid.flatten() if hasattr(ax_grid, "flatten") else [ax_grid]

                    ax = axs[panel_i % PAGE_CAP]
                    x  = np.log10(dwork["Quantity"]); yv = dwork["Cq"]
                    ax.scatter(x, yv, s=10, color="black")
                    xx = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
                    ax.plot(xx, model.predict(xx), "--", linewidth=0.8, color="black")
                    ax.set_title(f"{det}\nslope={slope:.3f}, R²={r2:.3f}", fontsize=8)
                    ax.set_xlabel("log10(Quantity)", fontsize=7)
                    ax.set_ylabel("Cq", fontsize=7)
                    ax.tick_params(labelsize=7)
                    for spine in ax.spines.values():
                        spine.set_linewidth(0.4)
                    panel_i += 1

                if fig is not None:
                    used = panel_i % PAGE_CAP or PAGE_CAP
                    if used < PAGE_CAP:
                        for k in range(used, PAGE_CAP):
                            axs[k].axis("off")
                    fig.tight_layout(pad=0.8); pdf.savefig(fig)
                    buf_png = io.BytesIO()
                    fig.savefig(buf_png, format="png", dpi=200, bbox_inches="tight")
                    std_page_pngs.append(buf_png.getvalue())
                    plt.close(fig)

        # C) CSV export (table)
        buf_csv = io.StringIO()
        st.session_state.df_smp_updated.to_csv(buf_csv, index=False)

        # D) ZIP (grid PDF + std PDF + CSV)
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(f"{base_name}_grid2x2.pdf", relq_pdf_buf.getvalue())
            zf.writestr(f"{base_name}_standard_curves.pdf", std_pdf_buf.getvalue())
            zf.writestr(f"{base_name}.csv", buf_csv.getvalue())

        st.download_button(
            "📦 Download results (ZIP)",
            data=zip_buf.getvalue(),
            file_name=f"{base_name}.zip",
            mime="application/zip"
        )

        # E) On-screen previews (same pages as PDFs)
        st.subheader("📄 Relative expression (grid) preview")
        if relq_page_pngs:
            for i, png in enumerate(relq_page_pngs, start=1):
                st.image(png, caption=f"RelQ grid page {i}", use_column_width=True)
        else:
            st.info("There are no pages available for display in the RelQ grid.")

        st.subheader("📄 STANDARD curves preview")
        if std_page_pngs:
            for i, png in enumerate(std_page_pngs, start=1):
                st.image(png, caption=f"STANDARD curves page {i}", use_column_width=True)
        else:
            st.info("There are no pages available for display in the STANDARD Curves section.")
