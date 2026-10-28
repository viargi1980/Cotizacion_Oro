import subprocess
import sys
from pathlib import Path

# ==========================================
# SCRIPTS
# ==========================================

scripts = [
    "main.py",
    "train.py",
    "prediccion.py",
    
]

# ==========================================
# EJECUCION
# ==========================================

for script in scripts:

    print()
    print("=" * 60)
    print(f"Ejecutando {script}")
    print("=" * 60)

    if not Path(script).exists():

        print(
            f"❌ No existe: {script}"
        )

        sys.exit(1)

    resultado = subprocess.run(
        [sys.executable, script]
    )

    if resultado.returncode != 0:

        print()
        print(
            f"❌ Error en {script}"
        )

        sys.exit(
            resultado.returncode
        )

    print()
    print(
        f"✅ {script} finalizado"
    )

# ==========================================
# FIN
# ==========================================

print()
print("=" * 60)
print("PIPELINE COMPLETADO")
print("=" * 60)

print()
print("✅ Dataset generado")
print("✅ Modelo entrenado")
print("✅ Predicción realizada")