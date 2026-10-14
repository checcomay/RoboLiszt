# RoboLiszt - tutti i comandi di sviluppo e di esecuzione. `make` (o `make help`) mostra l'elenco.
# Variabili modificabili da riga di comando, es.:  make task2 FINGER=ring     make sim SOUND=1 SOUND_BACKEND=wav
SHELL := /bin/bash
.DEFAULT_GOAL := help

ROS_DISTRO    ?= jazzy
ROS_WS        ?= $(HOME)/ros2_ws
VENV          ?= .venv
PKG           := roboliszt_gazebo
FINGER        ?= middle
PROFILE       ?= pp
RTF           ?= 1.0
RATE          ?= 100
SOUND         ?= 0
SOUND_BACKEND ?= sounddevice
SOUND_OUT     ?= piano_session.wav
FILE          ?= data/trajectories/task3_chord_C4E4G4.npz

# ogni riga di ricetta gira in una shell nuova: ambienti da ricaricare a ogni comando
PYENV    := if [ -f $(VENV)/bin/activate ]; then source $(VENV)/bin/activate; fi
ROS_BASE := source /opt/ros/$(ROS_DISTRO)/setup.bash
ROS_RUN  := $(ROS_BASE) && source $(ROS_WS)/install/setup.bash && export ROBOLISZT_ROOT=$(CURDIR)
SOUND_ARG := $(if $(filter 1 true yes,$(SOUND)),true,false)

# traiettorie dei task (data/trajectories/*.npz, prodotte da scripts/run_tasks.py)
TASK2_thumb  := task2_press_thumb_C4
TASK2_index  := task2_press_index_D4
TASK2_middle := task2_press_middle_E4
TASK2_ring   := task2_press_ring_F4
TASK2_little := task2_press_little_G4

runner = $(PYENV); $(ROS_RUN) && ros2 run $(PKG) task_runner $(1) --rate $(RATE)

.PHONY: help setup setup-sound link assets verify check-urdf build rebuild clean-ws \
        sim sim-effort task5 task task1 task2 task3 task3-roll task4 controllers topics record \
        test tasks-py analysis task5-py all-py kill clean clean-gen

help: ## questo elenco
	@awk 'BEGIN{FS=":.*## "} /^[a-zA-Z0-9_-]+:.*## /{printf "  \033[1m%-13s\033[0m %s\n",$$1,$$2}' $(MAKEFILE_LIST)
	@echo; echo "Variabili: FINGER=$(FINGER) PROFILE=$(PROFILE) RTF=$(RTF) SOUND=$(SOUND) SOUND_BACKEND=$(SOUND_BACKEND) ROS_WS=$(ROS_WS)"
	@echo "Due terminali: 'make sim' nel primo, poi 'make task1|task2|task3|task3-roll|task4' nel secondo."

# ------------------------------------------------------------------ ambiente
setup: ## crea .venv e installa le dipendenze Python
	python3 -m venv $(VENV) --system-site-packages
	$(PYENV); pip install numpy scipy pyyaml matplotlib pytest catkin_pkg empy lark

setup-sound: ## installa sounddevice (serve PortAudio: sudo apt install libportaudio2)
	$(PYENV); pip install sounddevice

link: ## collega gazebo/ nel workspace ROS (una volta sola)
	mkdir -p $(ROS_WS)/src
	ln -sfn $(CURDIR)/gazebo $(ROS_WS)/src/$(PKG)

# ------------------------------------------------------------------ asset e build
assets: ## rigenera URDF, mondo, mesh, controllers da config/robot.yaml (RTF=0.25 per il Task 5)
	$(PYENV); python3 gazebo/generate.py --rtf $(RTF)

verify: ## verifica indipendente URDF vs modello DH
	$(PYENV); python3 gazebo/verify_urdf.py

check-urdf: ## check_urdf + gz sdf -k su URDF e mondo (serve ROS)
	$(ROS_BASE) && check_urdf gazebo/urdf/roboliszt.urdf && gz sdf -k gazebo/worlds/piano.sdf

build: link assets verify ## rigenera gli asset e compila il pacchetto ROS 2
	$(PYENV); $(ROS_BASE) && cd $(ROS_WS) && colcon build --packages-select $(PKG) --symlink-install

clean-ws: ## cancella build/install/log del pacchetto nel workspace
	rm -rf $(ROS_WS)/build/$(PKG) $(ROS_WS)/install/$(PKG) $(ROS_WS)/log

rebuild: clean-ws build ## ricompila tutto da zero (dopo modifiche a URDF/mesh/mondo/YAML)

# ------------------------------------------------------------------ simulazione (terminale 1)
sim: ## Gazebo + controlli in modalita' posizione, Task 1-4 (SOUND=1 per il suono)
	$(PYENV); $(ROS_RUN) && ros2 launch $(PKG) roboliszt.launch.py sound:=$(SOUND_ARG) sound_backend:=$(SOUND_BACKEND) sound_out:=$(SOUND_OUT)

sim-effort: ## Gazebo in modalita' effort, senza avviare il tocco
	$(PYENV); $(ROS_RUN) && ros2 launch $(PKG) roboliszt.launch.py mode:=effort sound:=$(SOUND_ARG) sound_backend:=$(SOUND_BACKEND) sound_out:=$(SOUND_OUT)

task5: ## Task 5 completo in un solo terminale (RTF 0.25, FINGER=, PROFILE=pp|mf|ff)
	$(MAKE) build RTF=0.25
	$(PYENV); $(ROS_RUN) && ros2 launch $(PKG) roboliszt.launch.py mode:=effort touch:=true finger:=$(FINGER) profile:=$(PROFILE) sound:=$(SOUND_ARG) sound_backend:=$(SOUND_BACKEND) sound_out:=$(SOUND_OUT)

# ------------------------------------------------------------------ task (terminale 2)
data/trajectories/%.npz:
	$(PYENV); python3 scripts/run_tasks.py

task: ## invia una traiettoria qualsiasi: make task FILE=data/trajectories/xxx.npz
	$(call runner,$(FILE))

task1: data/trajectories/task1_wave.npz ## Task 1: saluto
	$(call runner,$<)

task2: data/trajectories/$(TASK2_$(FINGER)).npz ## Task 2: tasto singolo (FINGER=thumb|index|middle|ring|little)
	$(call runner,$<)

task3: data/trajectories/task3_chord_C4E4G4.npz ## Task 3: accordo C-E-G
	$(call runner,$<)

task3-roll: data/trajectories/task3_chord_C4E4G4_roll.npz ## Task 3: accordo arpeggiato
	$(call runner,$<)

task4: data/trajectories/task4_octave_jump_7.npz ## Task 4: salto d'ottava
	$(call runner,$<)

# ------------------------------------------------------------------ diagnostica
controllers: ## elenca i controller attivi
	$(ROS_RUN) && ros2 control list_controllers

topics: ## frequenza di /joint_states e stato dei tasti
	$(ROS_RUN) && (timeout 5 ros2 topic hz /joint_states; ros2 topic echo /world/piano_world/model/piano/joint_state --once)

record: ## registra un rosbag in bag_<ora> (Ctrl-C per fermare)
	$(ROS_RUN) && ros2 bag record -o bag_$$(date +%H%M%S) /joint_states /world/piano_world/model/piano/joint_state

kill: ## chiude Gazebo e i nodi rimasti attivi
	-pkill -f "gz sim"; pkill -f parameter_bridge; pkill -f robot_state_publisher; pkill -f ros2_control_node; pkill -f piano_sound_node; pkill -f task5_touch_node; true

# ------------------------------------------------------------------ solo Python (senza ROS)
test: assets ## esegue la suite di test
	$(PYENV); pytest tests/

tasks-py: ## Task 1-5 cinematici -> data/trajectories, metrics.json
	$(PYENV); python3 scripts/run_tasks.py

analysis: tasks-py ## ellissoidi, dinamica inversa, statica -> data/analysis.json
	$(PYENV); python3 scripts/run_analysis.py

task5-py: ## Task 5 a corpi rigidi (~2 min) -> data/task5_rigid.json
	$(PYENV); python3 scripts/run_task5_rigid.py

all-py: test tasks-py analysis task5-py ## tutta la pipeline Python

# ------------------------------------------------------------------ pulizia
clean: ## cancella cache Python
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

clean-gen: ## cancella i file generati da gazebo/generate.py (si rigenerano con `make assets`)
	rm -f gazebo/urdf/*.urdf gazebo/worlds/piano.sdf gazebo/config/controllers*.yaml gazebo/meshes/*.obj