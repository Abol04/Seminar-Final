import pandas as pd
from pyomo.environ import (
    ConcreteModel, Set, Var, Binary, Constraint, Objective, maximize, SolverFactory, inequality
)
import os

# --- 1. Einlesen der Daten ---
students = pd.read_excel("studierende.xlsx", sheet_name="Alle Studierende")
unis = pd.read_excel("unis.xlsx", sheet_name="Alle Universitäten")

# Universitäten mit aktuellem Aufnahmestopp entfernen
unis = unis[unis['Status'] != 'Pausiert']

# --- 2. Bewertung der Studierenden ---
weights = {
    'Note': 0.6,
    'Motivation': 0.2,
    'Sprache': 0.1,
    'Lebenslauf': 0.05,
    'Formalien': 0.05,
}

def normalize_note(note):
    return (5.0 - note) / 4.0

def normalize_motivation(motivation):
    return (3 - motivation) / 2.0

def normalize_sprache(sprache):
    mapping = {'C2': 1.0, 'C1': 0.8, 'B2': 0.6, 'B1': 0.4, 'A2': 0.2, 'A1': 0.0}
    return mapping.get(sprache, 0.0)

def normalize_lebenslauf(lebenslauf):
    return (3 - lebenslauf) / 2.0

def normalize_formalien(formalien):
    return (3 - formalien) / 2.0

def calculate_score(row):
    score = 0
    score += weights['Note'] * normalize_note(row['Note'])
    score += weights['Motivation'] * normalize_motivation(row['Motivation'])
    score += weights['Sprache'] * normalize_sprache(row['Sprache'])
    score += weights['Lebenslauf'] * normalize_lebenslauf(row['Lebenslauf'])
    score += weights['Formalien'] * normalize_formalien(row['Formalien'])
    return score

students['Score'] = students.apply(calculate_score, axis=1)

# Besondere Chancen & Malus
students.loc[
    (students['BesondereChance:Behinderung'] == True) | (students['BesondereChance:Kind'] == True),
    'Score'
] += 0.6
students.loc[
    (students['Level'] == 'Bachelor') & (students['ECTS'] < 60),
    'Score'
] -= 0.2

# Präferenzgewichtung für Gasthochschule (optional)
pref_weights = {1: 1.0, 2: 0.9, 3: 0.8, 4: 0.7, 5: 0.6}
for pref in range(1, 6):
    students[f'Score_Pref{pref}'] = students['Score'] / pref_weights.get(pref, 1.0) * pref_weights[pref]

# --- 3. Pyomo Modell ---
student_ids = students['Matrikelnummer'].tolist()
uni_ids = unis['ArbeitsnameUni'].tolist()

model = ConcreteModel()
model.students = Set(initialize=student_ids)
model.unis = Set(initialize=uni_ids)
model.x = Var(model.students, model.unis, domain=Binary)

# Jeder Student genau eine Uni (nur wenn überhaupt eine möglich ist)
def one_uni_per_student_rule(m, student):
    return sum(m.x[student, university] for university in m.unis) <= 1

model.one_uni_per_student = Constraint(model.students, rule=one_uni_per_student_rule)

# Kapazitätsbedingungen
#def bachelor_capacity_rule(m, u):
#    max_bachelor = unis.loc[unis['ArbeitsnameUni'] == u, 'MaxBachelor'].values[0]
#    if max_bachelor == 0:
#        # Keine Bachelor-Studierenden zulassen
#        return sum(m.x[s, u] for s in m.students if students.loc[students['Matrikelnummer'] == s, 'Level'].values[0] == 'Bachelor') == 0
#    else:
#        return sum(m.x[s, u] for s in m.students if students.loc[students['Matrikelnummer'] == s, 'Level'].values[0] == 'Bachelor') <= max_bachelor
#    
#model.bachelor_capacity = Constraint(model.unis, rule=bachelor_capacity_rule)

#def master_capacity_rule(m, u):
#    max_master = unis.loc[unis['ArbeitsnameUni'] == u, 'MaxMaster'].values[0]
#    if max_master == 0:
#        # Keine Master-Studierenden zulassen
#        return sum(m.x[s, u] for s in m.students if students.loc[students['Matrikelnummer'] == s, 'Level'].values[0] == 'Master') == 0
#    else:
#        return sum(m.x[s, u] for s in m.students if students.loc[students['Matrikelnummer'] == s, 'Level'].values[0] == 'Master') <= max_master
#model.master_capacity = Constraint(model.unis, rule=master_capacity_rule)

#def both_capacity_rule(m, u):
#    max_both = unis.loc[unis['ArbeitsnameUni'] == u, 'MaxBeide'].values[0]
#    return sum(m.x[s, u] for s in m.students) <= max_both
#model.both_capacity = Constraint(model.unis, rule=both_capacity_rule)

# Gleichverteilung WiSe/SoSe falls nötig
#def gleiche_aufteilung_rule(m, u):
#    if unis.loc[unis['ArbeitsnameUni'] == u, 'GleicheAufteilungWISESOSE'].values[0]:
#        wise = sum(
#            m.x[s, u]
#            for s in m.students
#            if students.loc[students['Matrikelnummer'] == s, 'Semesterwahl'].values[0] == 'WISE'
#        )
#        sose = sum(
#            m.x[s, u]
#            for s in m.students
#            if students.loc[students['Matrikelnummer'] == s, 'Semesterwahl'].values[0] == 'SOSE'
#        )
#        # 'Egal' students are not counted for balancing
#        diff = wise - sose
#        return inequality(-1, diff, 1)
#    else:
#        return Constraint.Feasible
#model.gleiche_aufteilung = Constraint(model.unis, rule=gleiche_aufteilung_rule)

# Programm-Match-Constraint (nur Unis mit passendem Programm)
#def programm_match_rule(m, s, u):
#    student_program = students.loc[students['Matrikelnummer'] == s, 'Programm'].values[0]
#    colname = f'Programm-{str(student_program).zfill(2)}'
#    if colname not in unis.columns:
#        return m.x[s, u] == 0
#    offers_program = unis.loc[unis['ArbeitsnameUni'] == u, colname].values[0]
#    return m.x[s, u] <= int(offers_program)
#model.programm_match = Constraint(model.students, model.unis, rule=programm_match_rule)

# Ziel: Maximierung des Gesamtnutzwerts
model.objective = Objective(
    expr=sum(
        students.loc[students['Matrikelnummer'] == s, 'Score'].values[0] * model.x[s, u]
        for s in model.students for u in model.unis
    ),
    sense=maximize
)

# --- 4. Lösen ---
solver = SolverFactory('highs')
result = solver.solve(model)
print("Solver Status:", result.solver.status)
print("Termination Condition:", result.solver.termination_condition)

import traceback

try:
    # --- 5. Zuweisungen extrahieren ---
    zuweisungen = []
    for s in model.students:
        for u in model.unis:
            val = model.x[s, u].value
            if val is not None and val > 0.5:
                zuweisungen.append({
                    'Matrikelnummer': s,
                    'Universität': u,
                    'Score': students.loc[students['Matrikelnummer'] == s, 'Score'].values[0]
                })

    if zuweisungen:
        print("Zuweisungen gefunden:", len(zuweisungen))
        print("Erste 5 Zuweisungen:", zuweisungen[:5])
        zuweisungen_df = pd.DataFrame(zuweisungen)
        zuweisungen_df.to_csv('zuweisungen.csv', index=False)
        print("Datei gespeichert unter:", os.path.abspath('zuweisungen.csv'))
    else:
        print("Keine Zuweisungen gefunden.")
except Exception as e:
    print("Fehler beim Extrahieren der Zuweisungen:", e)
    traceback.print_exc()