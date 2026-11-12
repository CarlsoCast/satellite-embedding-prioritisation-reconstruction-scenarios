"""Entrada vigente: ejecutar downstream limitado y reconstruir el mismo cuaderno."""
from threadpoolctl import threadpool_limits
import linea3_downstream_espectral as downstream
import linea3_discusion_espectral as discussion
from construir_cuaderno_ndmi import main as build
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        downstream.main()
        discussion.main()
        build()

    import runpy
    from pathlib import Path
    runpy.run_path(str(Path(__file__).with_name("ejecutar_cuaderno_linea3.py")),run_name="__main__")
