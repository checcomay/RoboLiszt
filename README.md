# RoboLiszt - mano a 20 DoF: cinematica, dinamica, controllo, 5 task pianistici

```
config/robot.yaml            parametri (DH, limiti, tastiera, controllo, carrello, corpi rigidi, attuatori)
src/hand_robot/
  model/                     dh.py, robot.py                          (invariati)
  kinematics/                jacobian.py (analitico, sul TCP), ik.py (DLS multi-catena, limiti, vincoli unilaterali),
                             singularity.py, manipulability.py (ellissoidi di velocita', forza, dinamica)
  dynamics/                  model.py + simulation.py (modello semplificato, + simulate_closed_loop),
                             reduced.py (modello ridotto polso + 1 dito, verificato contro quello completo),
                             rigid_body.py (Newton-Eulero ricorsivo su albero, M, g, h, energia)
  piano/keyboard.py          geometria tastiera dei task (solo tasti bianchi); piano/layout.py: tastiera completa C3-C6 con tasti neri (MIDI)
  audio/                     synth.py (sintesi additiva), detector.py (nota = tasto, velocity = velocita' di discesa), mixer.py, from_tasks.py
  visualization/hand_mesh.py mesh .obj della mano (visuale; collisioni e inerzie invariate)
  planning/trajectory.py     quintica a minimo jerk
  control/                   hybrid.py (posizione/forza, modello semplificato), statics.py (tau = J^T F),
                             opspace.py (controllo nello spazio operativo sul modello a corpi rigidi),
                             touch_node_core.py (nucleo del nodo ROS del Task 5, senza dipendenze ROS)
  tasks/                     task1_wave ... task5_touch, common.py (Scene)
  visualization/             figures.py (invariato), task_plots.py
gazebo/                      pacchetto ROS 2 ament_python `roboliszt_gazebo` (NON ESEGUITO: ROS 2/Gazebo non disponibili)
                             generate.py (URDF, mondo SDF, controllers.yaml [200 Hz] e controllers_effort.yaml [1 kHz]),
                             verify_urdf.py (verifica indipendente, ESEGUITO), launch/roboliszt.launch.py,
                             roboliszt_gazebo/task_runner.py (Task 1-4), roboliszt_gazebo/task5_touch_node.py (Task 5)
scripts/run_tasks.py         task -> data/trajectories/*.npz, data/metrics.json, docs/images/tasks
scripts/run_analysis.py      ellissoidi, dinamica inversa, statica, validazione IK -> data/analysis.json, docs/images/analysis
tests/                       test_roboliszt.py (16), test_task5_rigid.py (10), test_touch_node_core.py (3)
scripts/run_task5_rigid.py   Task 5 a corpi rigidi -> data/task5_rigid.json, data/trajectories/task5rb_*.npz, docs/images/task5_rigid
latex/                       sezione_controllo_spazio_operativo.tex, limiti_controllore_sostitutivo.tex
```
Ordine: `python scripts/run_tasks.py && python scripts/run_analysis.py && python gazebo/generate.py && python gazebo/verify_urdf.py`

Stato Gazebo/ROS 2: i file in gazebo/ sono stati scritti ma NON eseguiti (nessun ROS 2/Gazebo disponibile). Verificati solo:
URDF vs modello DH (verify_urdf.py, 1e-9), XML ben formato, sintassi Python (py_compile), nucleo del nodo Task 5 in anello chiuso su impianto
ridotto. Requisiti emersi: il controllo di forza richiede >= 1 kHz (data/task5_rigid_rate_sweep.json: a 200 Hz e' instabile);
il nodo Python costa ~2-5 ms per passo (usare generate.py --rtf 0.25); a fondo corsa la forza non e' ricostruibile dal solo stato del giunto.

Grafica e suono (aggiunti dopo): mesh della mano in gazebo/meshes/, tastiera con 22 tasti bianchi + 15 neri (key_<MIDI>_joint, C4 = key_60),
piano_sound_node (ros2 launch ... sound:=true sound_backend:=sounddevice|aplay|wav). Eseguiti solo offline: test (tests/test_audio.py,
test_hand_mesh.py, test_layout_world.py), anteprime docs/images/hand/, WAV dei task in data/audio/. Mesh, mondo con tasti neri e nodo del suono
NON sono stati eseguiti in Gazebo/ROS 2 da questa sessione. Con --rtf < 1 i backend in tempo reale suonano a tempo di parete: usare `wav`.

---

## Aggiornamento: dinamica unificata e studio di controllo (ultima sessione)

**Modello dinamico unico.** Tutte le analisi dinamiche e di controllo usano ora il modello a corpi rigidi (`dynamics/rigid_body.py`, completo; `dynamics/reduced.py`, ridotto a polso + un dito). Il modello con inerzia uniforme (`dynamics/model.py`, `dynamics/simulation.py`, `control/hybrid.py::HybridTouchController`, `tasks/task5_touch.py`) resta nel progetto **solo come riferimento storico** (confronto nella figura `docs/images/task5_rigid/task5_rigid_force_comparison.png`): sovrastima l'inerzia efficace della punta (~0,8 kg contro decine di grammi) e produce picchi di forza irrealistici. La relazione usa soltanto il modello unificato.

**Nuovi/modificati**
* `control/baselines.py` (nuovo): `JointSpaceController` (coppia calcolata nello spazio dei giunti) e `MotionController` (molla-smorzatore cartesiana con nucleo dinamicamente consistente; forma "accel" a pulsazione naturale o "force" in N/m = impedenza; opzione `zero_vref_on_contact`).
* `scripts/run_control_study.py` (nuovo): studio del capitolo di controllo, a parti riprendibili (`--only E1,E2a,...`, poi `--merge --plot`); scrive `data/control_study.json` e `docs/images/control/*.png` (etichette in italiano). Ogni parte dura meno di ~2 minuti.
* `config/robot.yaml`: `touch_rigid.wn_xy` 40 → **100** rad/s (con 40 il polpastrello deviava lateralmente di 8–11 mm; con 100 di 0,7–1,5 mm).
* `tasks/task5_rigid.py`: loggate le coordinate x, y del polpastrello; nuove metriche `lateral_max_mm`, `lateral_rms_mm`.
* `scripts/run_analysis.py`: la statica del contatto usa le traiettorie del Task 5 a dinamica unificata (`task5rb_*`); etichette in italiano.
* `scripts/run_tasks.py`: `--task N` ora **unisce** le metriche in `data/metrics.json` invece di sovrascriverlo.
* `visualization/task_plots.py`: etichette in italiano.
* `tests/test_baselines.py` (nuovo, 2 test): il controllo di sola posizione dipende dall'altezza del tasto e l'ibrido no; l'impedenza con rilevamento del contatto ha picco inferiore a quella semplice. Totale: **57 test**.

**Risultati principali dello studio (modello unificato)**: errore RMS in spazio libero 1,03 mm (giunti) / 0,96 mm (operativo) a parità di banda; con ±2 mm sull'altezza del tasto la forza a regime va da 1,97 a 4,32 N (posizione), 2,36–2,73 N (impedenza), 2,53 N costante (ibrido, anche con errore di gravità del 30%, rigidezza del tasto ±30%, indice e anulare). Il tocco ff richiede ~0,19 N·m al giunto del dito (limite nominale ipotizzato 0,1 N·m).

**Ricostruibilità**: `python scripts/run_task5_rigid.py` (~2,5 min), poi le parti di `run_control_study.py`. Gazebo/ROS 2: nessuna modifica; il pacchetto resta **non eseguito** in Gazebo.
