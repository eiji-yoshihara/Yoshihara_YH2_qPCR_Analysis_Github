import io, os, zipfile, warnings, csv
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.set_page_config(page_title="Yoshihara Lab SOP Software YH#2 qPCR_Analysis_v0.1.2", layout="wide")
st.title("🧬 Yoshihara Lab SOP Software YH#2 qPCR_Analysis_v0.1.2")

# ---------- Core helpers ----------
def read_qpcr_textfile(content_bytes: bytes) -> pd.DataFrame:
    # 文字コード: utf-8 → cp932 の順で試す
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
    header_idx = next((i for i,l in enumerate(lines) if l.strip().startswith("Well")), None)
    if header_idx is None:
        raise ValueError("Header line 'Well ...' Not Found")
    data_str = "\n".join(lines[header_idx:])

    # 区切り推定
    try:
        df = pd.read_csv(io.StringIO(data_str), sep="\t", engine="python")
        if df.shape[1] <= 1:
            raise ValueError
    except Exception:
        try:
            sniff = csv.Sniffer().sniff(data_str.split("\n",1)[0])
            sep = sniff.delimiter
        except Exception:
            sep = r"\s+"
        df = pd.read_csv(io.StringIO(data_str), sep=sep, engine="python")

    if "Reporter" in df.columns:
        df = df[df["Reporter"].astype(str)=="SYBR"]

    keep = ["Well","Sample Name","Detector Name","Reporter","Task","Ct","Quantity"]
    df = df.loc[:, [c for c in keep if c in df.columns]]
    return df

def clean_dataframe_for_analysis(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy(); df.columns = [c.strip() for c in df.columns]
    def _rn(target, cond):
        if target in df.columns: return
        cands = [c for c in df.columns if cond(c)]
        if cands: df.rename(columns={cands[0]:target}, inplace=True)
    _rn("Reporter", lambda c:c.lower()=="reporter")
    _rn("Ct", lambda c:c.lower()=="ct")
    _rn("Sample Name", lambda c:c.lower() in ("sample","sample_name","samplename"))
    _rn("Detector Name", lambda c:"detector" in c.lower())
    _rn("Task", lambda c:c.lower()=="task")
    _rn("Quantity", lambda c:"quantity" in c.lower())

    # Undetermined → NaN
    df["Ct"] = pd.to_numeric(df["Ct"].replace({"Undetermined":np.nan,"undetermined":np.nan}), errors="coerce")
    df["Quantity"] = pd.to_numeric(df.get("Quantity", np.nan), errors="coerce")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if "Well" in df.columns:
            df["Well"] = pd.to_numeric(df["Well"], errors="ignore")
    return df

def compute_standard_curve(df_std: pd.DataFrame):
    d = df_std.replace([np.inf, -np.inf], np.nan).dropna(subset=["Ct","Quantity"]).copy()
    d = d[d["Quantity"] > 0]
    if len(d) < 2:
        return None
    X = np.log10(d["Quantity"].values).reshape(-1,1)
    y = d["Ct"].values
    model = LinearRegression().fit(X,y)
    r2 = r2_score(y, model.predict(X))
    return {"slope": float(model.coef_[0]), "intercept": float(model.intercept_), "r2": float(r2), "model": model}

def ct_to_quantity(ct, slope, intercept):
    if pd.isna(ct) or slope == 0 or np.isnan(slope):
        return np.nan
    return float(10 ** ((ct - intercept) / slope))

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

        need = {"Task", "Ct", "Detector Name"}
        miss = [c for c in need if c not in st.session_state.df_raw.columns]
        if miss:
            st.error(f"Missing required columns: {miss}")
        else:
            cols = [c for c in ["Detector Name", "Task", "SourceFile"] if c in st.session_state.df_raw.columns]
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
        df_std = df_std[df_std["Task"].astype(str).str.lower()=="standard"].dropna(subset=["Ct"]).copy()
        if "Well" in df_std.columns: df_std["Well"] = pd.to_numeric(df_std["Well"], errors="coerce")
        df_std = df_std.sort_values(["Detector Name","Quantity","Well"], na_position="last")
        drops = []
        for (det, qty), sub in df_std.groupby(["Detector Name","Quantity"], dropna=False):
            st.markdown(f"**{det} — Qty {qty}**  (ΔCt={sub['Ct'].max()-sub['Ct'].min():.2f})")
            show = sub[["Well","Sample Name","Ct"]].reset_index()
            idxs = st.multiselect("Drop rows", options=show["index"].tolist(),
                                  format_func=lambda i: f"Well {int(df_std.loc[i,'Well']) if pd.notna(df_std.loc[i,'Well']) else '?'} / {df_std.loc[i,'Sample Name']} (Ct={df_std.loc[i,'Ct']})",
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
            for det in st.session_state.df_std_clean["Detector Name"].dropna().unique():
                ddf = st.session_state.df_std_clean[st.session_state.df_std_clean["Detector Name"]==det]
                sc = compute_standard_curve(ddf)
                fig, ax = plt.subplots(figsize=(5,3.2))
                ax.set_title(f"Standard curve: {det}")
                ax.set_xlabel("log10(Quantity)"); ax.set_ylabel("Ct")
                if sc:
                    x = np.log10(ddf["Quantity"]); y = ddf["Ct"]
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
        df_smp = st.session_state.df_raw.copy()
        df_smp = df_smp[df_smp["Task"].astype(str).str.lower()=="unknown"].copy()
        if df_smp.empty:
            st.warning("No 'Unknown' samples were found in the uploaded data.")
        else:
            st.session_state.conditions = st.text_area(
                "Conditions (one per line)", value="\n".join(st.session_state.conditions), height=100
            ).splitlines()
            st.session_state.conditions = [c.strip() for c in st.session_state.conditions if c.strip()]
            df_smp = df_smp.sort_values(["Detector Name","Well"], na_position="last").reset_index(drop=True)
            if "Condition" not in df_smp.columns: df_smp["Condition"] = ""
            if "Replicate" not in df_smp.columns: df_smp["Replicate"] = ""
            edited = st.data_editor(
                df_smp[["Detector Name","Sample Name","Ct","Condition","Replicate"]],
                column_config={
                    "Condition": st.column_config.SelectboxColumn(options=st.session_state.conditions),
                    "Replicate": st.column_config.SelectboxColumn(options=["Rep1","Rep2","Rep3"])
                },
                use_container_width=True, num_rows="fixed", key="assign_editor"
            )
            if st.button("Save assignment"):
                df_smp.loc[:, ["Condition","Replicate"]] = edited[["Condition","Replicate"]].values
                missing = df_smp[(df_smp["Condition"]=="") | (df_smp["Replicate"]=="")]
                if not missing.empty:
                    st.error(f"{len(missing)} samples missing assignment.")
                else:
                    st.session_state.df_smp = df_smp
                    st.success("Assignments saved.")

# 5) Quantify（helpers を使用：Undetermined/欠損は NaN 扱い）
with t5:
    if st.session_state.df_smp is None or st.session_state.df_std_clean is None:
        st.info("Please Complete 2) & 4)")
    else:
        df_smp = st.session_state.df_smp.copy()
        df_smp["Quantity"] = np.nan

        # 各 Detector の標準曲線を helper で作成し、ct_to_quantity で Quantity 化
        for det in st.session_state.df_std_clean["Detector Name"].dropna().unique():
            dstd = st.session_state.df_std_clean[
                (st.session_state.df_std_clean["Detector Name"] == det) &
                (st.session_state.df_std_clean["Task"].astype(str).str.lower() == "standard")
            ].copy()

            sc = compute_standard_curve(dstd)
            if sc is None:
                st.warning(f"Detector '{det}': Not enough standard points or Quantity <= 0. Skipped.")
                continue

            slope, intercept = sc["slope"], sc["intercept"]
            mask = (df_smp["Detector Name"] == det)
            df_smp.loc[mask, "Quantity"] = df_smp.loc[mask, "Ct"].apply(
                lambda c: ct_to_quantity(c, slope, intercept)
            )
            # 数値誤差などの負値防止
            df_smp.loc[mask & (df_smp["Quantity"] < 0), "Quantity"] = np.nan

        detectors_for_ctrl = sorted(df_smp["Detector Name"].dropna().unique().tolist())
        if not detectors_for_ctrl:
            st.error("Detector Name was not found. Please check Upload/Assign")
        else:
            ctrl_det = st.selectbox("Control detector", detectors_for_ctrl, key="ctrl_det_select")

            if st.button("Run Relative Quantification"):
                # Control Quantity（0 や NaN は無効→ NaN）
                ctrl_df = (
                    df_smp[df_smp["Detector Name"] == ctrl_det][["Condition", "Replicate", "Quantity"]]
                    .rename(columns={"Quantity": "Ctrl_Quantity"})
                    .copy()
                )
                ctrl_df.loc[(ctrl_df["Ctrl_Quantity"] <= 0) | (ctrl_df["Ctrl_Quantity"].isna()), "Ctrl_Quantity"] = np.nan

                # Fallback: Condition 平均
                ctrl_cond_mean = (
                    ctrl_df.groupby("Condition", as_index=False)["Ctrl_Quantity"]
                    .mean()
                    .rename(columns={"Ctrl_Quantity": "Ctrl_Cond_Mean"})
                )

                df_temp = df_smp.copy()
                if "Relative Quantity" not in df_temp.columns:
                    df_temp["Relative Quantity"] = np.nan

                for det in df_temp["Detector Name"].dropna().unique():
                    mask = df_temp["Detector Name"] == det
                    ddet = (
                        df_temp.loc[mask, ["Condition", "Replicate", "Quantity"]]
                        .reset_index()
                        .rename(columns={"index": "orig_index"})
                        .sort_values("orig_index").reset_index(drop=True)
                    )
# 同じ Condition & Replicate を持つ Control の行を抽出
                    cond_rep = ddet[["Condition", "Replicate"]].drop_duplicates()
                    ctrl_sub = ctrl_df.merge(cond_rep, on=["Condition", "Replicate"], how="inner")
                    ctrl_sub = ctrl_sub.sort_values("orig_index").reset_index(drop=True)
 # 行数を比較
                    if len(ctrl_sub) == len(ddet):
                # ✅ 対応数が一致 → 横結合で安全に結合
                        merged = pd.concat([ddet, ctrl_sub["Ctrl_Quantity"]], axis=1)
                        merged["Used_Ctrl"] = merged["Ctrl_Quantity"]
                        merged["Relative Quantity"] = merged["Quantity"] / merged["Used_Ctrl"]
                    else:
                # ⚠️ 行数が異なる（technical replicates数不一致）
                        warnings.warn(
                        f"⚠️ Detector '{det}': Control and target replicate counts differ "
                        f"({len(ctrl_sub)} vs {len(ddet)}). Using Condition-level mean instead."
                        )

                    merged = ddet.merge(ctrl_df, on=["Condition","Replicate"], how="left")
                    if merged["Ctrl_Quantity"].isna().any():
                        merged = merged.merge(ctrl_cond_mean, on="Condition", how="left")
                        merged["Used_Ctrl"] = merged["Ctrl_Quantity"].fillna(merged["Ctrl_Cond_Mean"])
                    else:
                        merged["Used_Ctrl"] = merged["Ctrl_Quantity"]

                    invalid_den = merged["Used_Ctrl"].isna() | (merged["Used_Ctrl"] <= 0)
                    invalid_num = merged["Quantity"].isna() | (merged["Quantity"] < 0)
                    merged["Relative Quantity"] = np.where(
                        invalid_den | invalid_num,
                        np.nan,
                        merged["Quantity"] / merged["Used_Ctrl"]
                    )

                    # ここで technical replicate 内に 0 があった場合の平均から除外は、#6 の描画集計側で対応（NaN 化したものは平均・SEMから自然に除外）
                    df_temp.loc[merged["orig_index"], "Relative Quantity"] = merged["Relative Quantity"].values

                # Replicate 単位（Condition×Replicate）で Mean/SEM
                stats = (
                    df_temp.groupby(["Detector Name", "Condition", "Replicate"], observed=False)["Relative Quantity"]
                    .agg(["mean", "sem"]).reset_index()
                    .rename(columns={"mean": "RelQ_Mean", "sem": "RelQ_SEM"})
                )

                st.session_state.df_smp_updated = df_temp.merge(
                    stats, on=["Detector Name", "Condition", "Replicate"], how="left"
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
        # ---- 日付入りベース名 ----
        today = pd.Timestamp.now().strftime("%Y-%m-%d")
        base_name = f"{today}qPCR_Results"

        # ---- 描画ヘルパー（相対量：バー±SEM(0.25pt)＋Rep黒ドット）----
        def draw_relq_panel(ax, ddf, conds):
            vals = ddf[["Condition","Replicate","Relative Quantity"]].dropna()
            if vals.empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=8)
                ax.set_axis_off()
                return False

            rep_means = (
                vals.groupby(["Condition","Replicate"], observed=False)["Relative Quantity"]
                    .mean()
                    .reset_index(name="Rep_Mean")
            )
            cond_stats = (
                rep_means.groupby("Condition", observed=False)["Rep_Mean"]
                    .agg(Mean="mean", SEM=lambda s: s.std(ddof=1) / np.sqrt(s.count()))
                    .reindex(conds).reset_index()
            )
            if cond_stats["Mean"].notna().sum() == 0:
                ax.text(0.5, 0.5, "NaN", ha="center", va="center", fontsize=8)
                ax.set_axis_off()
                return False

            yerr = cond_stats["SEM"].to_numpy()
            ax.bar(
                cond_stats["Condition"], cond_stats["Mean"],
                yerr=yerr, capsize=2, alpha=0.65, linewidth=0.4,
                error_kw={"elinewidth": 0.25, "capthick": 0.25}
            )

            cond_to_x = {c:i for i, c in enumerate(cond_stats["Condition"])}
            rep_offset = {"Rep1": -0.12, "Rep2": 0.0, "Rep3": 0.12}
            for _, row in rep_means.iterrows():
                c = row["Condition"]
                x = cond_to_x.get(c, None)
                if x is None:
                    continue
                off = rep_offset.get(str(row["Replicate"]), 0.0)
                ax.scatter(x + off, row["Rep_Mean"], s=18, zorder=3, color="black")

            ax.set_ylabel("Relative Quantity", fontsize=8)
            ax.set_ylim(bottom=0)
            ax.tick_params(axis="x", labelrotation=45, labelsize=7)
            ax.tick_params(axis="y", labelsize=7)
            for spine in ax.spines.values():
                spine.set_linewidth(0.4)
            return True

        # ---- レイアウト設定（2in x 2in のタイル）----
        NCOLS, NROWS = 4, 3             # 1ページ12面
        PANEL_W, PANEL_H = 2.0, 2.0     # 各パネル 2in四方
        FIG_W, FIG_H = NCOLS*PANEL_W, NROWS*PANEL_H

        dets = st.session_state.df_smp_updated["Detector Name"].dropna().unique().tolist()
        conds = st.session_state.conditions

        # A) 相対量グリッド PDF + UIプレビュー
        relq_pdf_buf = io.BytesIO()
        relq_page_pngs = []
        with PdfPages(relq_pdf_buf) as pdf:
            panel_i = 0
            fig = None; axs = None
            for det in dets:
                if panel_i % (NCOLS*NROWS) == 0:
                    if fig is not None:
                        fig.tight_layout(pad=0.8); pdf.savefig(fig)
                        buf_png = io.BytesIO(); fig.savefig(buf_png, format="png", dpi=200, bbox_inches="tight")
                        relq_page_pngs.append(buf_png.getvalue()); plt.close(fig)
                    fig, ax_grid = plt.subplots(NROWS, NCOLS, figsize=(FIG_W, FIG_H))
                    axs = ax_grid.flatten()

                ax = axs[panel_i % (NCOLS*NROWS)]
                ddf = st.session_state.df_smp_updated[st.session_state.df_smp_updated["Detector Name"]==det].copy()
                draw_relq_panel(ax, ddf, conds)
                ax.set_title(det, fontsize=9)
                panel_i += 1

            if fig is not None:
                used = panel_i % (NCOLS*NROWS) or (NCOLS*NROWS)
                if used < (NCOLS*NROWS):
                    for k in range(used, NCOLS*NROWS):
                        axs[k].axis("off")
                fig.tight_layout(pad=0.8); pdf.savefig(fig)
                buf_png = io.BytesIO(); fig.savefig(buf_png, format="png", dpi=200, bbox_inches="tight")
                relq_page_pngs.append(buf_png.getvalue()); plt.close(fig)

        # B) Standardカーブ PDF + UIプレビュー
        std_pdf_buf = io.BytesIO()
        std_page_pngs = []
        with PdfPages(std_pdf_buf) as pdf:
            if st.session_state.df_std_clean is not None and not st.session_state.df_std_clean.empty:
                panel_i = 0
                fig = None; axs = None
                for det in st.session_state.df_std_clean["Detector Name"].dropna().unique():
                    ddf = st.session_state.df_std_clean[st.session_state.df_std_clean["Detector Name"]==det].copy()
                    dwork = ddf.replace([np.inf, -np.inf], np.nan).dropna(subset=["Ct","Quantity"])
                    dwork = dwork[dwork["Quantity"] > 0]
                    if len(dwork) < 2:
                        continue

                    X = np.log10(dwork["Quantity"].to_numpy()).reshape(-1,1)
                    y = dwork["Ct"].to_numpy()
                    model = LinearRegression().fit(X,y)
                    slope = float(model.coef_[0]); intercept = float(model.intercept_)
                    r2 = r2_score(y, model.predict(X))

                    if panel_i % (NCOLS*NROWS) == 0:
                        if fig is not None:
                            fig.tight_layout(pad=0.8); pdf.savefig(fig)
                            buf_png = io.BytesIO(); fig.savefig(buf_png, format="png", dpi=200, bbox_inches="tight")
                            std_page_pngs.append(buf_png.getvalue()); plt.close(fig)
                        fig, ax_grid = plt.subplots(NROWS, NCOLS, figsize=(FIG_W, FIG_H))
                        axs = ax_grid.flatten()

                    ax = axs[panel_i % (NCOLS*NROWS)]
                    x = np.log10(dwork["Quantity"]); yv = dwork["Ct"]
                    ax.scatter(x, yv, s=10, color="black")
                    xx = np.linspace(x.min(), x.max(), 100).reshape(-1,1)
                    ax.plot(xx, model.predict(xx), "--", linewidth=0.8, color="black")
                    ax.set_title(f"{det}\nslope={slope:.3f}, R²={r2:.3f}", fontsize=8)
                    ax.set_xlabel("log10(Quantity)", fontsize=7)
                    ax.set_ylabel("Ct", fontsize=7)
                    ax.tick_params(labelsize=7)
                    for spine in ax.spines.values():
                        spine.set_linewidth(0.4)
                    panel_i += 1

                if fig is not None:
                    used = panel_i % (NCOLS*NROWS) or (NCOLS*NROWS)
                    if used < (NCOLS*NROWS):
                        for k in range(used, NCOLS*NROWS):
                            axs[k].axis("off")
                    fig.tight_layout(pad=0.8); pdf.savefig(fig)
                    buf_png = io.BytesIO(); fig.savefig(buf_png, format="png", dpi=200, bbox_inches="tight")
                    std_page_pngs.append(buf_png.getvalue()); plt.close(fig)

        # C) CSV
        buf_csv = io.StringIO()
        st.session_state.df_smp_updated.to_csv(buf_csv, index=False)

        # D) ZIP（相対量グリッドPDF + StandardカーブPDF + CSV）
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

        # E) UI プレビュー（PDFと同じページをPNGで表示）
        st.subheader("📄 Relative expression (grid) preview")
        if relq_page_pngs:
            for i, png in enumerate(relq_page_pngs, start=1):
                st.image(png, caption=f"RelQ grid page {i}", use_column_width=True)
        else:
            st.info("There are no pages available for display in the RelQ grid.")

        st.subheader("📄 Standard curves preview")
        if std_page_pngs:
            for i, png in enumerate(std_page_pngs, start=1):
                st.image(png, caption=f"Standard curves page {i}", use_column_width=True)
        else:
            st.info("There are no pages available for display in the Standard Curves section.")
