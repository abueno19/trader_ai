#!/bin/bash
# filepath: scripts/on-stop.sh

# Activar el entorno virtual
source .venv/bin/activate

# Guardar las librerías instaladas en un archivo
pip freeze > requirements.txt

# Desactivar el entorno virtual
deactivate

# Borrar el entorno virtual
rm -rf .venv

echo "Librerías guardadas y entorno virtual eliminado"