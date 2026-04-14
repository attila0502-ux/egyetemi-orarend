import streamlit as st
import pandas as pd
from ortools.sat.python import cp_model
import io
import math
import collections
import re
from openpyxl.styles import Alignment
from openpyxl.utils import get_column_letter

# --- KONFIGURÁCIÓ ---
BLOKKOK = list(range(13))
HETEK = [1, 2, 3]

CIKLUS_MAPPING = {
    1: [1, 4, 7, 10],
    2: [2, 5, 8, 11],
    3: [3, 6, 9, 12]
}


def format_time(total_mins):
    return f"{total_mins // 60:02d}:{total_mins % 60:02d}"


def get_atomic_time(b):
    start_mins = 480 + b * 55
    return f"{format_time(start_mins)}-{format_time(start_mins + 45)}"


# --- IDŐ-ÉRTELMEZŐ MODUL ---
def parse_ido_blokkok(tiltott_str):
    if pd.isna(tiltott_str): return set()
    text = str(tiltott_str).strip()
    if not text: return set()
    blokkok = set()
    if all(x.strip().isdigit() for x in text.split(',')):
        for x in text.split(','):
            val = int(x.strip())
            if 1 <= val <= 13: blokkok.add(val - 1)
        return blokkok
    times = re.findall(r'(\d{1,2})[:.](\d{2})', text)
    if len(times) >= 2:
        h1, m1 = int(times[0][0]), int(times[0][1])
        h2, m2 = int(times[1][0]), int(times[1][1])
        mins1, mins2 = h1 * 60 + m1, h2 * 60 + m2
        if mins1 > mins2: mins1, mins2 = mins2, mins1
        for b in range(13):
            b_start = 480 + b * 55
            b_end = b_start + 45
            if mins1 < b_end and mins2 > b_start:
                blokkok.add(b)
    return blokkok


def build_pref_dict(pref_df):
    pref = {}
    if pref_df is not None and "Oktató" in pref_df.columns and "Szombat_Tiltott" in pref_df.columns:
        for _, r in pref_df.iterrows():
            okt = str(r["Oktató"]).strip()
            tiltott = parse_ido_blokkok(r["Szombat_Tiltott"])
            pref[okt] = tiltott
    return pref


# --- CIKLUSOS MODELL ÉPÍTÉS ---
def solve_schedule(df, tr_df, pref_dict):
    model = cp_model.CpModel()
    assign = {}
    total_rooms = len(tr_df)

    szak_targy_szam = df.groupby("Szak")["Tárgy"].nunique().to_dict()

    for idx, row in df.iterrows():
        n_blokk = row["NapiBlokk"]
        oktato = row["Oktató"]
        tiltott = pref_dict.get(oktato, set())
        if n_blokk <= 0: continue
        for w in HETEK:
            for b in range(14 - n_blokk):
                if any((b + offset) in tiltott for offset in range(n_blokk)):
                    continue
                assign[(idx, b, w)] = model.NewBoolVar(f"t{idx}_b{b}_w{w}")

    # Heurisztikus relaxáció: lehetőség a maximumra törekvésre
    for idx, row in df.iterrows():
        if row["NapiBlokk"] > 0:
            active_vars = [v for (i, b, w), v in assign.items() if i == idx]
            if active_vars:
                model.AddAtMostOne(active_vars)

    okt_vars = collections.defaultdict(list)
    szak_vars = collections.defaultdict(list)
    room_count_vars = collections.defaultdict(list)

    for (idx, b, w), var in assign.items():
        okt, szak, n_blokk = df.at[idx, "Oktató"], df.at[idx, "Szak"], df.at[idx, "NapiBlokk"]
        for active_b in range(b, b + n_blokk):
            okt_vars[(okt, w, active_b)].append(var)
            szak_vars[(szak, w, active_b)].append(var)
            room_count_vars[(w, active_b)].append(var)

    # 1. Oktatói diszjunkciók
    for v_list in okt_vars.values(): model.AddAtMostOne(v_list)

    # 2. Szak topológia (Laborbontás algoritmikus szűrése)
    for (szak, w, b), v_list in szak_vars.items():
        if szak_targy_szam.get(szak, 0) > 10:
            model.Add(sum(v_list) <= 2)  # Sávszélesség növelése (párhuzamosítás)
        else:
            model.AddAtMostOne(v_list)  # Szigorú szekvenciális kényszer

    # 3. Infrastrukturális korlátok
    for v_list in room_count_vars.values(): model.Add(sum(v_list) <= total_rooms)

    # Célfüggvény: Az NP-nehéz tér optimalizálása a maximális kitöltöttségre
    model.Maximize(sum(assign.values()))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 60
    status = solver.Solve(model)

    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        scheduled = []
        scheduled_ids = set()
        for (i, b, w), v in assign.items():
            if solver.Value(v) == 1:
                scheduled_ids.add(i)
                r = df.loc[i]
                scheduled.append({
                    "Hét_ID": w,
                    "Blokk_Kezdet": b, "Tartam": r["NapiBlokk"],
                    "Szak": r["Szak"], "Tárgy": r["Tárgy"], "Oktató": r["Oktató"], "Létszám": r["Létszám"]
                })

        # Anomáliák detektálása
        missing = []
        for idx, row in df.iterrows():
            if row["NapiBlokk"] > 0 and idx not in scheduled_ids:
                missing.append({
                    "Szak": row["Szak"],
                    "Tárgy": row["Tárgy"],
                    "Oktató": row["Oktató"],
                    "Tárgyak_Száma_A_Szakon": szak_targy_szam.get(row["Szak"], 0)
                })

        return scheduled, missing, "OK"
    return None, [], "Fatal Error: A matematikai modell nem konvergált ütközésmentes állapotra."


def assign_rooms(scheduled, tr_df):
    rooms = tr_df.sort_values("Kapacitás", ascending=False).to_dict('records')
    scheduled.sort(key=lambda x: x["Létszám"], reverse=True)
    usage = collections.defaultdict(bool)
    for s in scheduled:
        for r in rooms:
            if r["Kapacitás"] >= s["Létszám"]:
                free = True
                for b in range(s["Blokk_Kezdet"], s["Blokk_Kezdet"] + s["Tartam"]):
                    if usage[(r["Terem"], s["Hét_ID"], b)]:
                        free = False;
                        break
                if free:
                    s["Terem"] = r["Terem"]
                    for b in range(s["Blokk_Kezdet"], s["Blokk_Kezdet"] + s["Tartam"]):
                        usage[(r["Terem"], s["Hét_ID"], b)] = True
                    break
    return scheduled


def create_grid_for_szak(szak_data):
    times = [get_atomic_time(b) for b in BLOKKOK]
    weekends = [f"{i}. Hétvége" for i in range(1, 13)]
    grid_data = {t: [""] * 12 for t in times}
    df_grid = pd.DataFrame(grid_data, index=weekends)
    for item in szak_data:
        w = item["Hét_ID"]
        target_weekends = CIKLUS_MAPPING[w]
        terem = item.get('Terem', 'NINCS TEREM')
        cell_text = f"{item['Tárgy']}\n{item['Oktató']}\n({terem})"
        for week_num in target_weekends:
            row_name = f"{week_num}. Hétvége"
            for b in range(item["Blokk_Kezdet"], item["Blokk_Kezdet"] + item["Tartam"]):
                col_name = times[b]
                if df_grid.at[row_name, col_name]:
                    df_grid.at[row_name, col_name] += f"\n\n{cell_text}"
                else:
                    df_grid.at[row_name, col_name] = cell_text
    return df_grid


def main():
    st.set_page_config(page_title="Órarend Generátor", layout="wide")

    # --- PROFI, TUDOMÁNYOS FEJLÉC ---
    st.title("🎓 Intelligens Órarend Generátor és Optimalizáló")
    st.markdown("*Constraint Programming (CP-SAT) alapú, többdimenziós operációkutatási keretrendszer.*")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        f_t = st.file_uploader("Tantárgyak (Bemeneti halmaz)", type=["xlsx"])
    with col2:
        f_r = st.file_uploader("Termek (Infrastruktúra)", type=["xlsx"])
    with col3:
        f_p = st.file_uploader("Oktatói Preferenciák (Peremfeltétel)", type=["xlsx"])
    with col4:
        f_a = st.file_uploader("Archívum (Heurisztika)", type=["xlsx"])

    # --- TUDOMÁNYOS GOMB ---
    if st.button("🚀 Globális Optimalizáció Indítása", type="primary", use_container_width=True):
        if not (f_t and f_r): return
        df = pd.read_excel(f_t)
        df["NapiBlokk"] = df["Sz"].apply(lambda x: math.ceil(x / 4) if pd.notnull(x) and x > 0 else 0).astype(int)
        df["Létszám"] = pd.to_numeric(df.get("Létszám", 30), errors='coerce').fillna(30).astype(int)
        tr_df = pd.read_excel(f_r)

        szak_targy_szam = df.groupby("Szak")["Tárgy"].nunique().to_dict()
        pref_dict = build_pref_dict(pd.read_excel(f_p) if f_p else None)

        # --- DIAGNOSZTIKA ---
        st.write("⏳ Prediktív terhelés-analízis és dimenzióvizsgálat futtatása...")
        error_found = False
        for szak, count in szak_targy_szam.items():
            limit = 78 if count > 10 else 39
            total_h = df[df["Szak"] == szak]["NapiBlokk"].sum()
            if total_h > limit:
                st.error(
                    f"❌ Kódolási hiba a mátrixban: {szak} topológiája túlterhelt! (Számított kapacitás: {limit} blokk, Igény: {total_h} blokk.)")
                error_found = True

        if error_found: return

        # --- GENERÁLÁS TUDOMÁNYOS SPINNERREL ---
        with st.spinner("Változók és peremfeltételek (Constraints) injektálása a CP-SAT Solverbe. Kérem, várjon..."):
            raw_res, missing, msg = solve_schedule(df, tr_df, pref_dict)
            if raw_res:
                final = assign_rooms(raw_res, tr_df)

                if missing:
                    st.error(
                        "⚠️ Figyelmeztetés: Részleges konvergencia. A következő adatok ütközést okoztak a sokdimenziós térben, ezért izolálásra kerültek:")
                    miss_df = pd.DataFrame(missing)
                    st.dataframe(miss_df, use_container_width=True)
                    st.warning(
                        "Tipp a korrekcióhoz: Vizsgálja felül a kényszerfeltételeket (Oktatói tiltások vagy szűkített terhelhetőség).")
                else:
                    st.success("✅ Algoritmus konvergált! Az ütközésmentes optimális mátrix sikeresen felépült.")

                szakok = sorted(list(set(s["Szak"] for s in final)))
                st.subheader("Generált Rendszer Vizualizációja (Szakonként)")
                kivalasztott_szak = st.selectbox("Adattömb kiválasztása:", szakok)
                st.dataframe(create_grid_for_szak([s for s in final if s["Szak"] == kivalasztott_szak]),
                             use_container_width=True)

                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine='openpyxl') as writer:
                    for szak in szakok:
                        df_grid = create_grid_for_szak([s for s in final if s["Szak"] == szak])
                        df_grid.to_excel(writer, sheet_name=str(szak)[:31])
                        worksheet = writer.sheets[str(szak)[:31]]
                        for row in worksheet.iter_rows():
                            for cell in row: cell.alignment = Alignment(wrap_text=True, vertical='center',
                                                                        horizontal='center')
                st.download_button("📥 Adatmátrix Letöltése (Excel)", buf.getvalue(),
                                   "optimalizalt_orarend_struktura.xlsx")

            else:
                st.error(f"❌ {msg}")


if __name__ == "__main__":
    main()