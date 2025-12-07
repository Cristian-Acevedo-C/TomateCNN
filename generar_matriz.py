"""
Proyecto: Deteccion de enfermedades en hojas de tomate
Script:   generar_matriz.py
Autor:    Cristian Acevedo

Descripcion:
    Script "alias" / compatibilidad que simplemente llama al evaluador
    completo definido en `generar_matriz_full.py`.

    Se mantiene este archivo para no romper notas, comandos antiguos
    o referencias en el informe, pero TODA la logica principal vive en:

        generar_matriz_full.py

    Si quieres ver:
        - Matriz de confusion (TRAIN + VAL)
        - CSV con la matriz
        - Accuracy global y por clase
        - Log detallado en TXT

    ejecuta cualquiera de estos comandos:

        python generar_matriz_full.py
        python generar_matriz.py      # hace exactamente lo mismo
"""

from generar_matriz_full import main


def run():
    """
    Punto de entrada "amigable" por si alguien importa este modulo desde
    otro script. Internamente solo delega en main().
    """
    main()


if __name__ == "__main__":
    # Entrada directa desde linea de comandos.
    # Equivalente a ejecutar: python generar_matriz_full.py
    run()
