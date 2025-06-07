import pandas as pd
from pyomo.environ import *

# --- 1. Einlesen der Daten ---
students = pd.read_excel("studierende.xlsx", engine="openpyxl")
unis = pd.read_excel("unis.xlsx", engine="openpyxl")

# --- 2. Datenbereinigung ---
students.columns = students.columns.str.strip()
unis.columns = unis.columns.str.strip()

students["Note"] = pd.to_numeric(students["Note"], errors="coerce").fillna(5.0)
students["Sprache"] = students["Sprache"].fillna("").str.strip()
students["BesondereChance:Behinderung"] = students["BesondereChance:Behinderung"].astype(str).str.lower().isin(["wahr", "true", "1"])
students["Level"] = students["Level"].fillna("").str.strip()
students["ECTS"] = pd.to_numeric(students["ECTS"], errors="coerce").fillna(0)

unis["Status"] = unis["Status"].fillna("").str.strip().str.lower()
unis["MaxBachelor"] = pd.to_numeric(unis["MaxBachelor"], errors="coerce").fillna(0)
unis["MaxMaster"] = pd.to_numeric(unis["MaxMaster"], errors="coerce").fillna(0)
unis["MaxBeide"] = pd.to_numeric(unis["MaxBeide"], errors="coerce").fillna(0)

# --- 3. Score-Berechnung ---
def calc_score(row):
    note_score = max(0, 1 - (row["Note"] - 1) * 0.25) * 0.6
    lang_score = 0.1 if row["Sprache"] else 0.0
    bonus = 0.1 if row["BesondereChance:Behinderung"] else 0.0
    malus = -0.1 if row["Level"].startswith("B.") and row["ECTS"] < 60 else 0.0
    return note_score + lang_score + bonus + malus
students["Score"] = students.apply(calc_score, axis=1)

# --- 4. Modellaufbau ---
model = ConcreteModel()
students_ids = students["Matrikelnummer"].tolist()
unis_names = unis["Hochschule"].tolist()
prios = range(1, 6)

# Alle zulässigen Zuordnungspaare (nur dort, wo Präferenz tatsächlich gesetzt ist)
assignable = []
for sid in students_ids:
    for prio in prios:
        uni = students.loc[students["Matrikelnummer"] == sid, f"Gasthochschule {prio}. kurz"].iloc[0]
        if isinstance(uni, str) and uni.strip() and uni in unis_names:
            assignable.append( (sid, uni) )

assignable = list(set(assignable))  # doppelte raus

model.x = Var(assignable, domain=Binary)

# --- Zielfunktion ---
pref_weights = {1: 1.0, 2: 0.9, 3: 0.8, 4: 0.7, 5: 0.6}
def weight_for(sid, uni):
    for prio in prios:
        pname = f"Gasthochschule {prio}. kurz"
        if students.loc[students["Matrikelnummer"] == sid, pname].iloc[0] == uni:
            return pref_weights[prio]
    return 0.0

model.obj = Objective(
    expr = sum(
        students.loc[students["Matrikelnummer"] == sid, "Score"].iloc[0] * weight_for(sid, uni) * model.x[(sid, uni)]
        for (sid, uni) in assignable
    ),
    sense = maximize
)

# --- Jeder Studierende max. 1 Platz ---
def student_rule(model, sid):
    return sum(model.x[(sid, uni)] for uni in unis_names if (sid, uni) in assignable) <= 1
model.student_con = Constraint(students_ids, rule=student_rule)

# --- Kapazitäten beachten ---
def uni_cap_rule(model, uni):
    cap = unis.loc[unis["Hochschule"] == uni, ["MaxBachelor", "MaxMaster", "MaxBeide"]].sum(axis=1).iloc[0]
    status = unis.loc[unis["Hochschule"] == uni, "Status"].iloc[0]
    if status != "aktiv":
        return sum(model.x[(sid, uni)] for sid in students_ids if (sid, uni) in assignable) == 0
    return sum(model.x[(sid, uni)] for sid in students_ids if (sid, uni) in assignable) <= cap
model.uni_cap = Constraint(unis_names, rule=uni_cap_rule)

# --- Programm-Kompatibilität (optional, falls z. B. nur Master an bestimmten Unis) ---
def prog_rule(model, sid, uni):
    student_level = students.loc[students["Matrikelnummer"] == sid, "Level"].iloc[0]
    if student_level not in unis.columns:
        return Constraint.Skip  # Keine Information -> kein Zwang
    allowed = unis.loc[unis["Hochschule"] == uni, student_level]
    if not allowed.empty and allowed.iloc[0] != 1:
        return model.x[(sid, uni)] == 0
    return Constraint.Skip
model.prog = Constraint(assignable, rule=lambda m, sid, uni: prog_rule(m, sid, uni))

# --- Modell lösen ---
solver = SolverFactory("highs")
result = solver.solve(model)

# --- Status ausgeben ---
if result.solver.termination_condition != TerminationCondition.optimal:
    print("Fehler: Modell nicht optimal lösbar. Prüfe Daten, Kapazitäten und Präferenzlisten.")
else:
    # Ergebnisse extrahieren
    assignments = [
        {"Matrikelnummer": sid, "Universität": uni}
        for (sid, uni) in assignable
        if model.x[(sid, uni)].value is not None and round(model.x[(sid, uni)].value) == 1
    ]
    if not assignments:
        print("Keine Zuweisungen gefunden. Prüfe Restriktionen, Kapazitäten und Präferenzen!")
    else:
        pd.DataFrame(assignments).to_excel("zuweisungen_final_pyomo.xlsx", index=False)
        print("Fertig! Zuweisungen exportiert nach: zuweisungen_final_pyomo.xlsx")
