#!/bin/bash
# filepath: scripts/on-start.sh

# Crear el entorno virtual
python -m venv .venv

# Activar el entorno virtual
source .venv/bin/activate

# Instalar las dependencias
pip install -r requirements.txt

echo "Entorno virtual creado y dependencias instaladas"