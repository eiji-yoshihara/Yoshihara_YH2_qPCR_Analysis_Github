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

    df["Ct"] = pd.to_numeric(df["Ct"].replace({"Undetermined":np.nan,"undetermined":np.nan}), errors="coerce")
    df["Quantity"] = pd.to_numeric(df.get("Quantity", np.nan), errors="coerce")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if "Well" in df.columns:
            df["Well"] = pd.to_numeric(df["Well"], errors="ignore")
    return df

def compute_standard_curve(df_std: pd.DataFrame):
    d = df_std.dropna(subset=["Ct","Quantity"]).copy()
    if len(d) < 2: return None
    X = np.log10(d["Quantity"].values).reshape(-1,1)
    y = d["Ct"].values
    model = LinearRegression().fit(X,y)
    r2 = r2_score(y, model.predict(X))
    return {"slope": float(model.coef_[0]), "intercept": float(model.intercept_), "r2": float(r2), "model": model}

def ct_to_quantity(ct, slope, intercept):
    if slope == 0 or np.isnan(slope): return np.nan
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
        "📄 Upload qPCR result files (multiple files allowed / TXT, TSV, CSV formats)",
        type=["txt", "tsv", "csv"],
        accept_multiple_files=True,
        key="uploader_multi",
        help="You can export a split from the same run or load multiple files together."
    )

    col_l, col_r = st.columns([1,1])
    load_clicked = col_l.button("Load file(s)")
    clear_clicked = col_r.button("Clear")

    if clear_clicked:
        st.session_state.df_raw = None
        st.experimental_rerun()

    if load_clicked:
        if not ups:
            st.warning("No file selected.")
        else:
            df_list = []
            errs = []
            for up in ups:
                try:
                    # バイト列を安全に取得（readよりgetvalueが堅牢）
                    content_bytes = up.getvalue()
                    df_tmp = read_qpcr_textfile(content_bytes)
                    df_tmp = clean_dataframe_for_analysis(df_tmp)
                    df_tmp["SourceFile"] = up.name
                    df_list.append(df_tmp)
                except Exception as e:
                    errs.append(f"{up.name}: {e}")

            if errs:
                st.error("Failed to load some of the files. See details below.↓")
                st.code("\n".join(errs))

            if df_list:
                df = pd.concat(df_list, axis=0, ignore_index=True)
                df = df.dropna(how="all").reset_index(drop=True)
                st.session_state.df_raw = df
                st.success(f"Loaded {len(df_list)} file(s). Total rows = {len(df):,}")
            else:
                st.warning("No files could be read or recognized.")

    # プレビュー & 必須列チェック
    if st.session_state.get("df_raw") is not None:
        st.caption("Preview of the beginning (up to 30 lines)")
        st.dataframe(st.session_state.df_raw.head(30), use_container_width=True)

        need = {"Task", "Ct", "Detector Name"}
        miss = [c for c in need if c not in st.session_state.df_raw.columns]
        if miss:
            st.error(f"Required column(s) missing: {miss}")
        else:
            cols = [c for c in ["Detector Name", "Task", "SourceFile"] if c in st.session_state.df_raw.columns]
            if cols:
                with st.expander("Summary（Detector/Task/SourceFile）", expanded=False):
                    st.write(
                        st.session_state.df_raw[cols]
                        .value_counts()
                        .reset_index(name="count")
                    )

# 2) Clean Standards（各(Detector, Quantity)ごとに行チェック→削除）
with t2:
    if st.session_state.df_raw is None:
        st.info("Please Complete Upload")
    else:
        # --- 元データ整形 ---
        df_std = st.session_state.df_raw.copy()
        df_std = df_std[
            df_std["Task"].astype(str).str.lower() == "standard"
        ].dropna(subset=["Ct"]).copy()

        if "Well" in df_std.columns:
            # 数値化（失敗はNaNのまま）
            df_std["Well"] = pd.to_numeric(df_std["Well"], errors="coerce")

        # 表示順
        df_std = df_std.sort_values(["Detector Name", "Quantity", "Well"], na_position="last")

        # --- 各グループごとに expander でチェックボックス表示 ---
        drops = []  # 削除する index を集める
        for (det, qty), sub in df_std.groupby(["Detector Name", "Quantity"], dropna=False):
            ct_min, ct_max = sub["Ct"].min(), sub["Ct"].max()
            diff = float(ct_max - ct_min) if pd.notna(ct_min) and pd.notna(ct_max) else np.nan

            title = f"{det} : Quantity {qty}"
            if pd.notna(diff) and diff >= 1.5:
                title = f"⚠️ {title} (ΔCt={diff:.2f})"

            with st.expander(title, expanded=False):
                st.caption("Please check the row(s) you would like to delete")
                # 行ごとのチェックボックス
                sub_show = sub.reset_index()  # 'index' 列 = 元の行 index
                cols_to_show = ["Well", "Sample Name", "Ct"]
                st.dataframe(sub_show[cols_to_show + ["index"]].rename(columns={"index": "row_id"}),
                             use_container_width=True)

                # チェックボックスを縦に並べる（IDは元 index）
                for _, row in sub_show.iterrows():
                    lbl = f"{row.get('Sample Name','?')} (Ct={row.get('Ct')})"
                    ck_key = f"std_ck_{det}_{qty}_{int(row['index'])}"
                    checked = st.checkbox(lbl, key=ck_key, value=False)
                    if checked:
                        drops.append(int(row["index"]))

        # --- 確定ボタン ---
        if st.button("Apply cleaning", type="primary"):
            clean = df_std.drop(index=list(set(drops))).reset_index(drop=True)
            st.session_state.df_std_clean = clean
            st.success(f"Cleaned: {len(df_std)} → {len(clean)} rows "
                       f"({len(set(drops))} row(s) removed)")
        # プレビュー
        if st.session_state.get("df_std_clean") is not None:
            st.dataframe(st.session_state.df_std_clean.head(40), use_container_width=True)

# 3) Standard Curves（堅牢版：Quantity>0 & finite のみ使用、標準点>=2）
with t3:
    if st.session_state.get("df_std_clean") is None:
        st.info("2) Please do Clean Standards")
    else:
        buf_pdf = io.BytesIO()
        with PdfPages(buf_pdf) as pdf:

            # 既存のクリーン標準データから Detector ごとに作図
            for det in st.session_state.df_std_clean["Detector Name"].dropna().unique():
                raw = st.session_state.df_std_clean.copy()

                # 念のため Standard のみ・Ct/Quantity の有限値のみ・Quantity>0 に限定
                ddf = raw[raw["Detector Name"] == det].copy()
                ddf = ddf.replace([np.inf, -np.inf], np.nan)
                ddf = ddf.dropna(subset=["Ct", "Quantity"])
                ddf = ddf[ddf["Quantity"] > 0]

                fig, ax = plt.subplots(figsize=(6, 4))
                ax.set_title(f"Standard curve: {det}")
                ax.set_xlabel("log10(Quantity)")
                ax.set_ylabel("Ct")

                if len(ddf) >= 2:
                    try:
                        x = np.log10(ddf["Quantity"].to_numpy())
                        y = ddf["Ct"].to_numpy()

                        # 可視化（散布図）
                        ax.scatter(x, y, s=16, label="Data")

                        # 線形回帰
                        X = x.reshape(-1, 1)
                        model = LinearRegression().fit(X, y)
                        yhat = model.predict(X)
                        r2 = r2_score(y, yhat)
                        slope = float(model.coef_[0])
                        intercept = float(model.intercept_)

                        # フィット線
                        xx = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
                        yy = model.predict(xx)
                        ax.plot(xx.ravel(), yy, "--", linewidth=1.2, label="Fit")

                        # テキスト情報
                        ax.text(
                            0.02, 0.02,
                            f"n={len(ddf)}\nslope={slope:.3f}\nR²={r2:.3f}",
                            transform=ax.transAxes,
                            ha="left", va="bottom"
                        )

                        ax.legend(loc="best", fontsize=8)
                    except Exception as e:
                        ax.text(0.5, 0.5, f"Error in fit: {e}", ha="center")
                else:
                    ax.text(0.5, 0.5, "Insufficient standard points (n < 2)", ha="center")

                plt.tight_layout()
                st.pyplot(fig, clear_figure=True)
                pdf.savefig(fig)
                plt.close(fig)

        st.download_button(
            "📄 Download standard-curve report (PDF)",
            data=buf_pdf.getvalue(),
            file_name="qpcr_standard_curve_report.pdf",
            mime="application/pdf",
        )

# 4) Assign
with t4:
    if st.session_state.df_raw is None:
        st.info("Please Complete Upload")
    else:
        # ---- 元データ（Unknown のみ）を整える ----
        work = st.session_state.df_raw.copy()
        work = work[work["Task"].astype(str).str.lower() == "unknown"].copy()

        # 並び順：Detector→Well（数値にできる場合）
        if "Well" in work.columns:
            work["Well"] = pd.to_numeric(work["Well"], errors="coerce")
            work = work.sort_values(["Detector Name", "Well"], na_position="last").reset_index(drop=False)
            # 'index' 列が元のインデックス（この番号で assign_df を管理）
            base_index = work["index"].to_list()
        else:
            work = work.sort_values(["Detector Name"]).reset_index(drop=False)
            base_index = work["index"].to_list()

        # ---- セッション側の割り当てテーブル（Condition/Replicate）を用意 ----
        if "assign_df" not in st.session_state:
            st.session_state.assign_df = pd.DataFrame(index=st.session_state.df_raw.index)
            st.session_state.assign_df["Condition"] = ""
            st.session_state.assign_df["Replicate"] = ""
        # Detector テンプレ用
        if "detector_template" not in st.session_state:
            st.session_state.detector_template = None

        # ---- Condition 候補を編集 ----
        st.subheader("Define conditions")
        cond_text = st.text_area(
            "Conditions (1行に1つ)",
            value="\n".join(st.session_state.get("conditions", ["Control", "Treatment1", "Treatment2"])),
            height=100,
            help="Editing the options here will also update the selections in the dropdown box below."
        )
        st.session_state.conditions = [c.strip() for c in cond_text.splitlines() if c.strip()]

        st.markdown("---")

        # ---- Detector ごとの割り当て UI ----
        for det in sorted(work["Detector Name"].dropna().unique().tolist()):
            with st.expander(f"Detector: {det}", expanded=False):
                sub = work[work["Detector Name"] == det].copy().reset_index(drop=True)
                # sub には 'index' 列（元 df_raw の行番号）がある

                # ====== フィルター（Sample 名 / Well / Ct） ======
                f1, f2, f3 = st.columns([2, 1, 1])
                with f1:
                    q = st.text_input("Filter (Sample/Well/Ct 部分一致)", key=f"q_{det}").strip()
                with f2:
                    hide_nan = st.checkbox("Ct NaN を隠す", value=False, key=f"nan_{det}")
                with f3:
                    st.caption(f"Rows in this detector: {len(sub)}")

                filt = np.ones(len(sub), dtype=bool)
                if q:
                    qlow = q.lower()
                    filt &= (
                        sub["Sample Name"].astype(str).str.lower().str.contains(qlow) |
                        sub.get("Well", pd.Series([""] * len(sub))).astype(str).str.lower().str.contains(qlow) |
                        sub.get("Ct", pd.Series([""] * len(sub))).astype(str).str.lower().str.contains(qlow)
                    )
                if hide_nan and "Ct" in sub.columns:
                    filt &= sub["Ct"].notna()

                view = sub.loc[filt].copy()

                # ====== 対象行の複数選択 & 全選択 ======
                st.caption("Select samples to apply")
                # 表示ラベルと元インデックス（df_raw の行番号）をペアに
                options = [
                    (
                        f"{r['Sample Name']} (Well={'' if pd.isna(r.get('Well')) else int(r.get('Well')) if float(r.get('Well')).is_integer() else r.get('Well')}, Ct={r.get('Ct')})",
                        int(r["index"])
                    )
                    for _, r in view.iterrows()
                ]
                sel_key = f"ms_{det}"
                selected = st.multiselect(
                    "Samples",
                    [v for (_, v) in options],
                    format_func=lambda idx: next(lbl for (lbl, v) in options if v == idx),
                    key=sel_key
                )

                csel1, csel2 = st.columns([1, 1])
                if csel1.button("Select all (filtered)", key=f"selall_{det}"):
                    st.session_state[sel_key] = [v for (_, v) in options]
                    st.experimental_rerun()
                if csel2.button("Clear selection", key=f"selclr_{det}"):
                    st.session_state[sel_key] = []
                    st.experimental_rerun()

                # ====== Condition / Replicate を選び、選択行 or 全行に適用 ======
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

                # ====== Detector テンプレ（サイズ一致時のみ貼付け） ======
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

                # ====== プレビュー（現在の割当てを反映した表） ======
                st.caption("Updated table (after operations)")
                sub_preview = work[work["Detector Name"] == det][["index", "Sample Name", "Ct"]].copy()
                sub_preview["Condition"] = st.session_state.assign_df.loc[sub_preview["index"], "Condition"].values
                sub_preview["Replicate"] = st.session_state.assign_df.loc[sub_preview["index"], "Replicate"].values
                sub_preview = sub_preview.drop(columns=["index"])
                st.dataframe(sub_preview, use_container_width=True, hide_index=True)

        st.markdown("---")

        # ---- 保存（検証してセッションへ確定） ----
        if st.button("Save assignment", type="primary", key="save_assign"):
            assigned = st.session_state.assign_df.copy()
            # Unknown 対象だけを抽出
            target_idx = st.session_state.df_raw.index[
                st.session_state.df_raw["Task"].astype(str).str.lower() == "unknown"
            ]
            # 必須チェック
            missing_mask = (assigned.loc[target_idx, "Condition"] == "") | (assigned.loc[target_idx, "Replicate"] == "")
            if missing_mask.any():
                miss_count = int(missing_mask.sum())
                st.error(f"{miss_count} samples missing assignment (Condition or Replicate).")
            else:
                # 元データに列が無ければ作る
                if "Condition" not in st.session_state.df_raw.columns:
                    st.session_state.df_raw["Condition"] = ""
                if "Replicate" not in st.session_state.df_raw.columns:
                    st.session_state.df_raw["Replicate"] = ""

                # 反映
                st.session_state.df_raw.loc[:, "Condition"] = assigned["Condition"].fillna(st.session_state.df_raw["Condition"])
                st.session_state.df_raw.loc[:, "Replicate"] = assigned["Replicate"].fillna(st.session_state.df_raw["Replicate"])

                # 後続タブで使う Unknown の完成版を保存
                st.session_state.df_smp = st.session_state.df_raw[
                    st.session_state.df_raw["Task"].astype(str).str.lower() == "unknown"
                ].copy()

                st.success("Assignments saved.")

# 5) Quantify（Undetected は Quantity=0 として保持。ただし統計・描画では 0 を除外）
with t5:
    if st.session_state.df_smp is None or st.session_state.df_std_clean is None:
        st.info("Please Complete 2) & 4)")
    else:
        # Ct → Quantity（Ct 欠損/Undetected は 0 扱い）
        def _ct_to_qty(ct, slope, intercept):
            if pd.isna(ct):
                return 0.0
            if slope == 0 or np.isnan(slope):
                return np.nan
            return float(10 ** ((ct - intercept) / slope))

        # 1) 標準曲線から Quantity を付与
        df_smp = st.session_state.df_smp.copy()
        df_smp["Quantity"] = np.nan

        for det in st.session_state.df_std_clean["Detector Name"].dropna().unique():
            dstd = st.session_state.df_std_clean[
                (st.session_state.df_std_clean["Detector Name"] == det) &
                (st.session_state.df_std_clean["Task"].astype(str).str.lower() == "standard")
            ].copy()

            # 安全ガード
            dstd = dstd.replace([np.inf, -np.inf], np.nan).dropna(subset=["Ct", "Quantity"])
            dstd = dstd[dstd["Quantity"] > 0]
            if len(dstd) < 2:
                st.warning(f"'{det}': Standard curve calculation skipped: fewer than 2 standard points or Quantity ≤ 0.")
                continue

            X = np.log10(dstd["Quantity"].to_numpy()).reshape(-1, 1)
            y = dstd["Ct"].to_numpy()
            model = LinearRegression().fit(X, y)
            slope = float(model.coef_[0]); intercept = float(model.intercept_)

            rows_all = (df_smp["Detector Name"] == det)
            df_smp.loc[rows_all, "Quantity"] = df_smp.loc[rows_all, "Ct"].apply(
                lambda c: _ct_to_qty(c, slope, intercept)
            )
            # 非物理値は NaN
            df_smp.loc[rows_all & (df_smp["Quantity"] < 0), "Quantity"] = np.nan

        # 2) Control detector を選択して相対量を計算
        detectors_for_ctrl = sorted(df_smp["Detector Name"].dropna().unique().tolist())
        if not detectors_for_ctrl:
            st.error("Detector Name was not found. Please check Upload/Assign")
        else:
            ctrl_det = st.selectbox("Control detector", detectors_for_ctrl, key="ctrl_det_select")

            if st.button("Run Relative Quantification"):
                # Control の Quantity（分母候補）
                ctrl_df = (
                    df_smp[df_smp["Detector Name"] == ctrl_det][["Condition", "Replicate", "Quantity"]]
                    .rename(columns={"Quantity": "Ctrl_Quantity"})
                    .copy()
                )
                # 分母 0/NaN/負は無効（ゼロ割回避）
                ctrl_df.loc[
                    (ctrl_df["Ctrl_Quantity"].isna()) | (ctrl_df["Ctrl_Quantity"] <= 0),
                    "Ctrl_Quantity"
                ] = np.nan

                # Fallback: Condition 平均（>0 のみが平均に寄与）
                ctrl_cond_mean = (
                    ctrl_df.groupby("Condition", as_index=False)["Ctrl_Quantity"]
                    .mean()
                    .rename(columns={"Ctrl_Quantity": "Ctrl_Cond_Mean"})
                )

                # 作業テーブル
                df_temp = df_smp.copy()
                if "Relative Quantity" not in df_temp.columns:
                    df_temp["Relative Quantity"] = np.nan

                # 各 Detector について相対量を計算（同一 Condition & Replicate を対応付け）
                for det in df_temp["Detector Name"].dropna().unique():
                    mask = (df_temp["Detector Name"] == det)
                    ddet = (
                        df_temp.loc[mask, ["Condition", "Replicate", "Quantity"]]
                        .reset_index()
                        .rename(columns={"index": "orig_index"})
                    )

                    merged = ddet.merge(ctrl_df, on=["Condition", "Replicate"], how="left")

                    # Ctrl が欠けた行は Condition 平均で補完
                    if merged["Ctrl_Quantity"].isna().any():
                        merged = merged.merge(ctrl_cond_mean, on="Condition", how="left")
                        merged["Used_Ctrl"] = merged["Ctrl_Quantity"].fillna(merged["Ctrl_Cond_Mean"])
                    else:
                        merged["Used_Ctrl"] = merged["Ctrl_Quantity"]

                    # 分子も 0/NaN/負は無効（0 は「非検出」として除外）
                    invalid_den = merged["Used_Ctrl"].isna() | (merged["Used_Ctrl"] <= 0)
                    invalid_num = merged["Quantity"].isna() | (merged["Quantity"] <= 0)

                    merged["Relative Quantity"] = np.where(
                        invalid_den | invalid_num,
                        np.nan,
                        merged["Quantity"] / merged["Used_Ctrl"]
                    )

                    # 反映
                    df_temp.loc[merged["orig_index"], "Relative Quantity"] = merged["Relative Quantity"].values

                # ★ コントロール遺伝子は常に 1（Quantity>0 の行）。0/NaN行は NaN のまま
                df_temp.loc[
                    (df_temp["Detector Name"] == ctrl_det) & (df_temp["Quantity"] > 0),
                    "Relative Quantity"
                ] = 1.0

                # 3) 技術反復（Replicate）レベルの平均・SEM を作成
                #    → 0（非検出由来）は除外するため、Relative Quantity > 0 のみ採用
                relq_valid = df_temp[["Detector Name", "Condition", "Replicate", "Relative Quantity"]].copy()
                relq_valid = relq_valid.dropna(subset=["Relative Quantity"])
                relq_valid = relq_valid[relq_valid["Relative Quantity"] > 0]

                stats = (
                    relq_valid.groupby(["Detector Name", "Condition", "Replicate"], observed=False)["Relative Quantity"]
                    .agg(RelQ_Mean="mean",
                         RelQ_SEM=lambda s: s.std(ddof=1) / np.sqrt(s.count()) if s.count() > 1 else 0.0)
                    .reset_index()
                )

                # 元にマージ（無効セルは RelQ_Mean/SEM が欠損のまま）
                st.session_state.df_smp_updated = df_temp.merge(
                    stats, on=["Detector Name", "Condition", "Replicate"], how="left"
                )

                st.success("Relative quantification done.")
                # 参考：非 NaN の相対量カウント
                dbg = st.session_state.df_smp_updated.groupby("Detector Name")["Relative Quantity"].apply(
                    lambda s: int(s.notna().sum())
                )
                st.caption("Non-NaN Relative Quantity count per detector")
                st.write(dbg)

        # プレビュー
        if st.session_state.get("df_smp_updated") is not None:
            st.dataframe(st.session_state.df_smp_updated.head(30), use_container_width=True)

# 6) Export（PDF: 2in×2in グリッド & StandardカーブPDFも同梱 / UIにもプレビュー）
with t6:
    if st.session_state.df_smp_updated is None:
        st.info("Please proceed with 5)")
    else:
        import io, zipfile
        from matplotlib.backends.backend_pdf import PdfPages

        # ---- 日付入りベース名 ----
        today = pd.Timestamp.now().strftime("%Y-%m-%d")
        base_name = f"{today}qPCR_Results"

        # ---- 描画ヘルパー（相対量：バー±SEM＋Rep黒ドット）
        # 0 は統計から除外するが、描画は 0 バーを出す（カテゴリを消さない）
        def draw_relq_panel(ax, ddf, conds):
            vals = ddf[["Condition", "Replicate", "Relative Quantity"]].copy()

            # Repごとの平均（0 は統計から除外）
            rep_means = (
                vals[vals["Relative Quantity"] > 0]
                .groupby(["Condition", "Replicate"], observed=False)["Relative Quantity"]
                .mean()
                .reset_index(name="Rep_Mean")
            )

            # Conditionごとの Mean/SEM を算出。reindex で全条件を保持
            if not rep_means.empty:
                cond_stats = (
                    rep_means.groupby("Condition", observed=False)["Rep_Mean"]
                    .agg(Mean="mean", SEM=lambda s: s.std(ddof=1) / np.sqrt(s.count()))
                    .reindex(conds)
                )
            else:
                cond_stats = pd.DataFrame(index=conds, data={"Mean": np.nan, "SEM": np.nan})

            # 描画は 0 で埋めてバーを必ず表示
            plot_means = cond_stats["Mean"].fillna(0.0).to_numpy()
            plot_sems = cond_stats["SEM"].fillna(0.0).to_numpy()
            xlabels = cond_stats.index.tolist()

            ax.bar(
                xlabels, plot_means,
                yerr=plot_sems, capsize=2, alpha=0.65, linewidth=0.4,
                error_kw={"elinewidth": 0.25, "capthick": 0.25}
            )

            # 黒ドット（Rep_Mean）。0 は統計から除外したまま
            cond_to_x = {c: i for i, c in enumerate(xlabels)}
            rep_offset = {"Rep1": -0.12, "Rep2": 0.0, "Rep3": 0.12}
            for _, row in rep_means.iterrows():
                x = cond_to_x.get(row["Condition"], None)
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
        FIG_W, FIG_H = NCOLS * PANEL_W, NROWS * PANEL_H

        dets = st.session_state.df_smp_updated["Detector Name"].dropna().unique().tolist()
        conds = st.session_state.conditions

        # A) 相対量グリッド PDF + UIプレビュー
        relq_pdf_buf = io.BytesIO()
        relq_page_pngs = []
        with PdfPages(relq_pdf_buf) as pdf:
            panel_i = 0
            fig = None
            axs = None
            for det in dets:
                if panel_i % (NCOLS * NROWS) == 0:
                    if fig is not None:
                        fig.tight_layout(pad=0.8)
                        pdf.savefig(fig)
                        buf_png = io.BytesIO()
                        fig.savefig(buf_png, format="png", dpi=200, bbox_inches="tight")
                        relq_page_pngs.append(buf_png.getvalue())
                        plt.close(fig)
                    fig, ax_grid = plt.subplots(NROWS, NCOLS, figsize=(FIG_W, FIG_H))
                    axs = ax_grid.flatten()

                ax = axs[panel_i % (NCOLS * NROWS)]
                ddf = st.session_state.df_smp_updated[
                    st.session_state.df_smp_updated["Detector Name"] == det
                ].copy()
                draw_relq_panel(ax, ddf, conds)
                ax.set_title(det, fontsize=9)
                panel_i += 1

            # 最後のページを flush
            if fig is not None:
                used = panel_i % (NCOLS * NROWS) or (NCOLS * NROWS)
                if used < (NCOLS * NROWS):
                    for k in range(used, NCOLS * NROWS):
                        axs[k].axis("off")
                fig.tight_layout(pad=0.8)
                pdf.savefig(fig)
                buf_png = io.BytesIO()
                fig.savefig(buf_png, format="png", dpi=200, bbox_inches="tight")
                relq_page_pngs.append(buf_png.getvalue())
                plt.close(fig)

        # B) Standardカーブ PDF + UIプレビュー
        std_pdf_buf = io.BytesIO()
        std_page_pngs = []
        with PdfPages(std_pdf_buf) as pdf:
            if st.session_state.df_std_clean is not None and not st.session_state.df_std_clean.empty:
                panel_i = 0
                fig = None
                axs = None
                for det in st.session_state.df_std_clean["Detector Name"].dropna().unique():
                    ddf = st.session_state.df_std_clean[
                        st.session_state.df_std_clean["Detector Name"] == det
                    ].copy()
                    dwork = ddf.replace([np.inf, -np.inf], np.nan).dropna(subset=["Ct", "Quantity"])
                    dwork = dwork[dwork["Quantity"] > 0]
                    if len(dwork) < 2:
                        continue

                    X = np.log10(dwork["Quantity"].to_numpy()).reshape(-1, 1)
                    y = dwork["Ct"].to_numpy()
                    model = LinearRegression().fit(X, y)
                    slope = float(model.coef_[0])
                    intercept = float(model.intercept_)
                    r2 = r2_score(y, model.predict(X))

                    if panel_i % (NCOLS * NROWS) == 0:
                        if fig is not None:
                            fig.tight_layout(pad=0.8)
                            pdf.savefig(fig)
                            buf_png = io.BytesIO()
                            fig.savefig(buf_png, format="png", dpi=200, bbox_inches="tight")
                            std_page_pngs.append(buf_png.getvalue())
                            plt.close(fig)
                        fig, ax_grid = plt.subplots(NROWS, NCOLS, figsize=(FIG_W, FIG_H))
                        axs = ax_grid.flatten()

                    ax = axs[panel_i % (NCOLS * NROWS)]
                    x = np.log10(dwork["Quantity"])
                    yv = dwork["Ct"]
                    ax.scatter(x, yv, s=10, color="black")
                    xx = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
                    ax.plot(xx, model.predict(xx), "--", linewidth=0.8, color="black")
                    ax.set_title(f"{det}\nslope={slope:.3f}, R²={r2:.3f}", fontsize=8)
                    ax.set_xlabel("log10(Quantity)", fontsize=7)
                    ax.set_ylabel("Ct", fontsize=7)
                    ax.tick_params(labelsize=7)
                    for spine in ax.spines.values():
                        spine.set_linewidth(0.4)
                    panel_i += 1

                if fig is not None:
                    used = panel_i % (NCOLS * NROWS) or (NCOLS * NROWS)
                    if used < (NCOLS * NROWS):
                        for k in range(used, NCOLS * NROWS):
                            axs[k].axis("off")
                    fig.tight_layout(pad=0.8)
                    pdf.savefig(fig)
                    buf_png = io.BytesIO()
                    fig.savefig(buf_png, format="png", dpi=200, bbox_inches="tight")
                    std_page_pngs.append(buf_png.getvalue())
                    plt.close(fig)

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
