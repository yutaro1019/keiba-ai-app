"""
競馬データの前処理・特徴量エンジニアリング(超メモリ最適化版)
- 1パス読み込み + dtype縮小 + 履歴特徴は groupby.transform/cumsum で軽量化
"""
import pandas as pd
import numpy as np
import re
import os
import gc

DATA_DIR = "/home/user/datalist/extracted"
pd.set_option('future.no_silent_downcasting', True)


# ---------- パーサ群 ----------
def parse_race_info(s):
    if pd.isna(s):
        return (np.nan, None, "other", "?", np.nan, None, None)
    s = str(s).replace("\xa0", " ")
    m_round = re.match(r"\s*(\d+)\s*R", s)
    round_no = int(m_round.group(1)) if m_round else np.nan
    cls_match = re.search(r"R\s+(\S+?)\s", s)
    race_class = cls_match.group(1) if cls_match else None
    m2 = re.search(r"(芝|ダ|障)(右|左|直)?(\d+)m", s)
    if m2:
        surface = {"芝": "turf", "ダ": "dirt", "障": "jump"}.get(m2.group(1), "other")
        direction = m2.group(2) or "?"
        distance = int(m2.group(3))
    else:
        surface, direction, distance = "other", "?", np.nan
    w = re.search(r"天候\s*:\s*(\S+)", s)
    weather = w.group(1) if w else None
    g = re.search(r"(芝|ダート|障)\s*:\s*(\S+)", s)
    going = g.group(2) if g else None
    return (round_no, race_class, surface, direction, distance, weather, going)


def parse_sex_age(s):
    if pd.isna(s):
        return None, np.nan
    m = re.match(r"([牡牝セ騙])(\d+)", str(s))
    if m:
        return m.group(1), int(m.group(2))
    return None, np.nan


def parse_weight(s):
    if pd.isna(s) or str(s) in ("**", ""):
        return np.nan, np.nan
    m = re.match(r"\s*(\d+)\s*\(([+\-]?\d+)\)", str(s))
    if m:
        return int(m.group(1)), int(m.group(2))
    try:
        return int(str(s).strip()), 0
    except Exception:
        return np.nan, np.nan


def parse_passing(s):
    if pd.isna(s) or s in ("**", ""):
        return np.nan, np.nan, np.nan
    parts = re.findall(r"\d+", str(s))
    if not parts:
        return np.nan, np.nan, np.nan
    nums = [int(p) for p in parts]
    return nums[0], nums[-1], float(np.mean(nums))


def parse_chakujun(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s.isdigit():
        return int(s)
    return np.nan


# ---------- 1年分の読み込み(縮小済) ----------
def load_year(year):
    path = os.path.join(DATA_DIR, f"raw_keiba_data_{year}_full_final.csv")
    df = pd.read_csv(path, encoding="utf-8", low_memory=False)
    df.columns = [c.replace("\ufeff", "") for c in df.columns]
    df = df.rename(columns={
        'ﾀｲﾑ指数\n\n\nタイム指数(通常)\nタイム指数マスター': 'time_index',
    })
    keep = [c for c in [
        'race_id', '開催日', 'レース情報',
        '着順', '枠番', '馬番', '性齢', '斤量',
        '通過', '上り', '単勝', '人気', '馬体重',
        'horse_id', 'jockey_id', 'trainer_id',
        'time_index',
    ] if c in df.columns]
    df = df[keep].copy()

    # レース情報
    rinfo = df['レース情報'].apply(parse_race_info)
    df['round_no'] = np.array([x[0] for x in rinfo], dtype="float32")
    df['race_class'] = [x[1] for x in rinfo]
    df['surface'] = [x[2] for x in rinfo]
    df['direction'] = [x[3] for x in rinfo]
    df['distance'] = np.array([x[4] for x in rinfo], dtype="float32")
    df['weather'] = [x[5] for x in rinfo]
    df['going'] = [x[6] for x in rinfo]
    df.drop(columns=['レース情報'], inplace=True)

    sa = df['性齢'].apply(parse_sex_age)
    df['sex'] = [x[0] for x in sa]
    df['age'] = np.array([x[1] for x in sa], dtype="float32")
    df.drop(columns=['性齢'], inplace=True)

    bw = df['馬体重'].apply(parse_weight)
    df['body_weight'] = np.array([x[0] for x in bw], dtype="float32")
    df['body_weight_diff'] = np.array([x[1] for x in bw], dtype="float32")
    df.drop(columns=['馬体重'], inplace=True)

    df['rank'] = df['着順'].apply(parse_chakujun).astype("float32")
    df.drop(columns=['着順'], inplace=True)

    pas = df['通過'].apply(parse_passing)
    df['passing_first'] = np.array([x[0] for x in pas], dtype="float32")
    df['passing_last'] = np.array([x[1] for x in pas], dtype="float32")
    df['passing_mean'] = np.array([x[2] for x in pas], dtype="float32")
    df.drop(columns=['通過'], inplace=True)

    df['agari'] = pd.to_numeric(df['上り'], errors='coerce').astype("float32")
    df['odds'] = pd.to_numeric(df['単勝'], errors='coerce').astype("float32")
    df['popularity'] = pd.to_numeric(df['人気'], errors='coerce').astype("float32")
    df['weight_carry'] = pd.to_numeric(df['斤量'], errors='coerce').astype("float32")
    df['frame_no'] = pd.to_numeric(df['枠番'], errors='coerce').astype("float32")
    df['horse_no'] = pd.to_numeric(df['馬番'], errors='coerce').astype("float32")
    df.drop(columns=['上り', '単勝', '人気', '斤量', '枠番', '馬番'], inplace=True)

    if 'time_index' in df.columns:
        df['time_index'] = pd.to_numeric(
            df['time_index'].replace({"**": np.nan, "": np.nan}), errors='coerce'
        ).astype("float32")

    df['date'] = pd.to_datetime(df['開催日'], errors='coerce')
    df.drop(columns=['開催日'], inplace=True)

    # venue: race_idの5-6文字目
    df['venue'] = df['race_id'].astype("int64").astype(str).str.slice(4, 6)

    # ID列はint32に圧縮(NaNが無いidのみ)
    for c in ['horse_id', 'jockey_id', 'trainer_id']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').astype("Int32")

    df['race_id'] = df['race_id'].astype("int64")

    # object列をcategoryに(メモリ節約)
    for c in ['race_class', 'surface', 'direction', 'weather', 'going', 'sex', 'venue']:
        if c in df.columns:
            df[c] = df[c].astype("category")

    gc.collect()
    return df


# ---------- 血統 ----------
def extract_first_word(cell):
    if pd.isna(cell):
        return None
    s = str(cell).split("(")[0].split("[")[0]
    s = re.split(r"\s+", s.strip())[0]
    return s if s else None


def load_pedigree_features():
    path = os.path.join(DATA_DIR, "horse_pedigree_archive.csv")
    p = pd.read_csv(path, encoding="utf-8",
                    usecols=lambda c: c in ('horse_id', 'p1', 'p25'),
                    low_memory=False)
    out = pd.DataFrame()
    out['horse_id'] = pd.to_numeric(p['horse_id'], errors='coerce').astype("Int32")
    out['sire'] = p['p1'].apply(extract_first_word) if 'p1' in p.columns else None
    out['broodmare_sire'] = p['p25'].apply(extract_first_word) if 'p25' in p.columns else None
    return out


# ---------- 履歴特徴量(超軽量版) ----------
def add_history_features(df: pd.DataFrame, verbose=True) -> pd.DataFrame:
    """
    cumsum/cumcount ベースで expanding を使わず履歴特徴量を計算(高速・低メモリ)。
    すべて「過去のみ」を見るためにshiftしてから累積。
    """
    df = df.sort_values(["date", "race_id", "horse_no"], kind="mergesort").reset_index(drop=True)
    n = len(df)
    df["field_size"] = df.groupby("race_id")["horse_no"].transform("count").astype("float32")
    if verbose:
        print(f"    sorted, n={n:,}", flush=True)

    def cum_mean_past(values_shifted, group):
        """shift済み値の過去平均を、cumsum/cumcount で計算"""
        v = values_shifted.fillna(0).astype("float32").values
        valid = (~values_shifted.isna()).astype("float32").values
        cs = pd.Series(v).groupby(group).cumsum().values
        cn = pd.Series(valid).groupby(group).cumsum().values
        with np.errstate(invalid='ignore', divide='ignore'):
            out = np.where(cn > 0, cs / cn, np.nan).astype("float32")
        return out

    def cum_max_past(values_shifted, group):
        return values_shifted.groupby(group).cummax().astype("float32").values

    def add_rank_rates(prefix, keys):
        """過去のみの着順から、条件別の勝率/3着内率を作る。"""
        grp = df.groupby(keys, sort=False, observed=True)
        df[f"{prefix}_runs"] = grp.cumcount().astype("float32")
        rank_shifted = grp["rank"].shift()
        group_values = [df[k] for k in keys] if isinstance(keys, list) else df[keys]
        valid = ~rank_shifted.isna()
        df[f"{prefix}_win_rate"] = cum_mean_past((rank_shifted == 1).astype("float32").where(valid), group_values)
        df[f"{prefix}_top3_rate"] = cum_mean_past((rank_shifted <= 3).astype("float32").where(valid), group_values)
        del rank_shifted

    # ---- 馬の履歴 ----
    if verbose:
        print("    horse history...", flush=True)
    hg = df["horse_id"]
    df["horse_runs"] = df.groupby(hg).cumcount().astype("float32")

    rank_shift = df.groupby(hg)["rank"].shift()
    df["horse_avg_rank"] = cum_mean_past(rank_shift, hg)
    df["horse_win_rate"] = cum_mean_past((rank_shift == 1).astype("float32").where(~rank_shift.isna()), hg)
    df["horse_top3_rate"] = cum_mean_past((rank_shift <= 3).astype("float32").where(~rank_shift.isna()), hg)
    del rank_shift; gc.collect()

    if "time_index" in df.columns:
        ti_shift = df.groupby(hg)["time_index"].shift()
        df["horse_avg_time_idx"] = cum_mean_past(ti_shift, hg)
        df["horse_best_time_idx"] = cum_max_past(ti_shift, hg)
        del ti_shift; gc.collect()

    agari_shift = df.groupby(hg)["agari"].shift()
    df["horse_avg_agari"] = cum_mean_past(agari_shift, hg)
    del agari_shift; gc.collect()

    df["days_since_last"] = df.groupby(hg)["date"].diff().dt.days.astype("float32")

    if verbose:
        print("    recent form/running style...", flush=True)
    for lag in (1, 2, 3):
        df[f"horse_rank_lag{lag}"] = df.groupby(hg)["rank"].shift(lag).astype("float32")
        if "time_index" in df.columns:
            df[f"horse_time_idx_lag{lag}"] = df.groupby(hg)["time_index"].shift(lag).astype("float32")
        df[f"horse_agari_lag{lag}"] = df.groupby(hg)["agari"].shift(lag).astype("float32")
        df[f"horse_passing_first_lag{lag}"] = df.groupby(hg)["passing_first"].shift(lag).astype("float32")
        df[f"horse_passing_last_lag{lag}"] = df.groupby(hg)["passing_last"].shift(lag).astype("float32")
        df[f"horse_field_size_lag{lag}"] = df.groupby(hg)["field_size"].shift(lag).astype("float32")
        df[f"horse_distance_lag{lag}"] = df.groupby(hg)["distance"].shift(lag).astype("float32")

        denom = (df[f"horse_field_size_lag{lag}"] - 1).replace(0, np.nan)
        df[f"horse_passing_first_rate_lag{lag}"] = ((df[f"horse_passing_first_lag{lag}"] - 1) / denom).astype("float32")
        df[f"horse_passing_last_rate_lag{lag}"] = ((df[f"horse_passing_last_lag{lag}"] - 1) / denom).astype("float32")
        df[f"horse_passing_gain_lag{lag}"] = (df[f"horse_passing_first_lag{lag}"] - df[f"horse_passing_last_lag{lag}"]).astype("float32")
        df[f"horse_distance_diff_lag{lag}"] = (df["distance"].astype("float32") - df[f"horse_distance_lag{lag}"]).astype("float32")

    df["horse_recent_avg_rank3"] = df[["horse_rank_lag1", "horse_rank_lag2", "horse_rank_lag3"]].mean(axis=1).astype("float32")
    df["horse_recent_top3_rate3"] = (df[["horse_rank_lag1", "horse_rank_lag2", "horse_rank_lag3"]] <= 3).mean(axis=1).astype("float32")
    df["horse_front_style"] = df[["horse_passing_first_rate_lag1", "horse_passing_first_rate_lag2", "horse_passing_first_rate_lag3"]].mean(axis=1).astype("float32")
    df["horse_closing_style"] = df[["horse_passing_gain_lag1", "horse_passing_gain_lag2", "horse_passing_gain_lag3"]].mean(axis=1).astype("float32")
    df["horse_shorter_than_last"] = (df["horse_distance_diff_lag1"] < 0).astype("float32")
    df["horse_longer_than_last"] = (df["horse_distance_diff_lag1"] > 0).astype("float32")
    df["closer_short_distance_risk"] = (df["horse_closing_style"].clip(lower=0) * (df["distance"] <= 1400).astype("float32")).astype("float32")
    df["front_short_distance_fit"] = ((1 - df["horse_front_style"].clip(lower=0, upper=1)) * (df["distance"] <= 1400).astype("float32")).astype("float32")
    df["closer_long_distance_fit"] = (df["horse_closing_style"].clip(lower=0) * (df["distance"] >= 1800).astype("float32")).astype("float32")
    df["front_long_distance_risk"] = ((1 - df["horse_front_style"].clip(lower=0, upper=1)) * (df["distance"] >= 1800).astype("float32")).astype("float32")

    # ---- 騎手の履歴 ----
    if verbose:
        print("    jockey history...", flush=True)
    jg = df["jockey_id"]
    df["jockey_runs"] = df.groupby(jg).cumcount().astype("float32")
    jr_shift = df.groupby(jg)["rank"].shift()
    df["jockey_win_rate"] = cum_mean_past((jr_shift == 1).astype("float32").where(~jr_shift.isna()), jg)
    df["jockey_top3_rate"] = cum_mean_past((jr_shift <= 3).astype("float32").where(~jr_shift.isna()), jg)
    del jr_shift; gc.collect()

    # ---- 調教師の履歴 ----
    if verbose:
        print("    trainer history...", flush=True)
    tg = df["trainer_id"]
    tr_shift = df.groupby(tg)["rank"].shift()
    df["trainer_win_rate"] = cum_mean_past((tr_shift == 1).astype("float32").where(~tr_shift.isna()), tg)
    df["trainer_top3_rate"] = cum_mean_past((tr_shift <= 3).astype("float32").where(~tr_shift.isna()), tg)
    del tr_shift; gc.collect()

    # ---- 距離・コース適性 ----
    if verbose:
        print("    distance/surface adaptation...", flush=True)
    add_rank_rates("horse_dist", ["horse_id", "distance"])
    add_rank_rates("horse_surface", ["horse_id", "surface"])
    add_rank_rates("horse_venue", ["horse_id", "venue"])
    add_rank_rates("horse_course", ["horse_id", "venue", "surface", "distance"])
    add_rank_rates("jockey_venue", ["jockey_id", "venue"])
    add_rank_rates("jockey_course", ["jockey_id", "venue", "surface", "distance"])
    add_rank_rates("trainer_venue", ["trainer_id", "venue"])
    add_rank_rates("trainer_course", ["trainer_id", "venue", "surface", "distance"])

    return df


# ---------- 全部まとめてロード ----------
def load_all_data(years=range(2017, 2026), with_history=True, with_pedigree=True, verbose=True):
    parts = []
    for y in years:
        if verbose:
            print(f"  loading {y}...", flush=True)
        df_y = load_year(y)
        parts.append(df_y)
        gc.collect()
    df = pd.concat(parts, ignore_index=True)
    del parts
    gc.collect()
    if verbose:
        print(f"  total rows: {len(df):,}, mem: {df.memory_usage(deep=True).sum()/1e6:.1f}MB", flush=True)

    if with_pedigree:
        if verbose:
            print("  merging pedigree...", flush=True)
        ped = load_pedigree_features()
        df = df.merge(ped, on="horse_id", how="left")
        for c in ['sire', 'broodmare_sire']:
            if c in df.columns:
                df[c] = df[c].astype("category")
        del ped
        gc.collect()
        if verbose:
            print(f"  after pedigree merge: mem={df.memory_usage(deep=True).sum()/1e6:.1f}MB", flush=True)

    if with_history:
        if verbose:
            print("  building history features...", flush=True)
        df = add_history_features(df, verbose=verbose)
        gc.collect()
        if verbose:
            print(f"  after history: mem={df.memory_usage(deep=True).sum()/1e6:.1f}MB", flush=True)

    return df


if __name__ == "__main__":
    df = load_all_data(years=[2024], with_history=True, with_pedigree=True)
    print(df.shape)
    print(df[["race_id", "rank", "horse_avg_rank", "jockey_win_rate", "sire"]].head(8))
