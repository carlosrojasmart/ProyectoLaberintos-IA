import numpy as np
from collections import deque
import heapq
import time

# Función que lee una matriz de adyacencia desde un archivo de texto
def leer_matriz_adyacencia(ruta_archivo):
    # Abrir el archivo y leer las dimensiones de la matriz
    with open(ruta_archivo, 'r') as archivo:
        # La primera línea contiene las dimensiones (ancho, alto)
        dimensiones = archivo.readline().strip().strip('(').strip(')').split(',')
        ancho = int(dimensiones[0])
        alto = int(dimensiones[1])
        
        if ancho != alto:
            raise ValueError('La matriz debe ser cuadrada')
                
        matriz = []
        nodo_inicio = None
        nodo_meta = None
        
        # Leer el contenido del archivo línea por línea para construir la matriz
        for i, linea in enumerate(archivo):
            # Convertir la línea en una lista de enteros, eliminando corchetes y separando por comas
            fila = list(map(int, linea.strip().strip('[]').split(',')))
            matriz.append(fila)
            
            # Identificar el nodo de inicio (valor 2 en la matriz)
            if 2 in fila:
                nodo_inicio = (i, fila.index(2))
            
            # Identificar el nodo meta (valor 3 en la matriz)
            if 3 in fila:
                nodo_meta = (i, fila.index(3))
        
        # Construir la lista de adyacencia como un diccionario
        lista_adyacencia = {}
        
        # Verificar si la matriz es cuadrada (mismo número de filas y columnas)
        if len(matriz) != len(matriz[0]):
             raise ValueError('La matriz debe ser cuadrada')

        
        # Iterar sobre cada celda de la matriz
        for i in range(alto):
            for j in range(ancho):
                # Solo considerar las celdas que no sean muros (valor diferente a 1)
                if matriz[i][j] != 1:
                    nodo = (i, j)  # Definir el nodo actual
                    
                    adyacentes = []  # Lista para los nodos adyacentes
                    
                    # Verificar si la celda superior es accesible
                    if i > 0 and matriz[i-1][j] != 1:
                        adyacentes.append(((i-1, j), 1))
                    
                    # Verificar si la celda inferior es accesible
                    if i < alto-1 and matriz[i+1][j] != 1:
                        adyacentes.append(((i+1, j), 1))
                        
                    # Verificar si la celda a la izquierda es accesible
                    if j > 0 and matriz[i][j-1] != 1:
                        adyacentes.append(((i, j-1), 1))
                        
                    # Verificar si la celda a la derecha es accesible
                    if j < ancho-1 and matriz[i][j+1] != 1:
                        adyacentes.append(((i, j+1), 1))
                    
                    # Agregar el nodo y sus adyacentes a la lista de adyacencia
                    lista_adyacencia[nodo] = adyacentes
        
        # Retornar la lista de adyacencia, el nodo de inicio y el nodo meta
        return lista_adyacencia, nodo_inicio, nodo_meta


# Clase para representar un grafo a partir de la lista de adyacencia
class Grafo:
    def __init__(self, lista_adyacencia):
        self.lista_adyacencia = lista_adyacencia

    # Obtener los vecinos de un nodo específico
    def obtener_vecinos(self, v):
        return self.lista_adyacencia.get(v, [])
    
    # Función heurística: distancia Manhattan
    def h(self, n, nodo_meta):
        return abs(n[0] - nodo_meta[0]) + abs(n[1] - nodo_meta[1])


    # Implementación de búsqueda en profundidad (DFS)
    def primero_profundidad(self, nodo_inicio, nodo_final, visitados=None, camino=None):
        # Inicializar el conjunto de nodos visitados y el camino si es la primera llamada
        if visitados is None:
            visitados = set()
        if camino is None:
            camino = []

        # Marcar el nodo como visitado y agregarlo al camino actual
        visitados.add(nodo_inicio)
        camino.append(nodo_inicio)

        # Si se alcanza el nodo final, retornar el camino encontrado
        if nodo_inicio == nodo_final:
            return camino
        
        # Recorrer los vecinos del nodo actual
        for vecino, _ in self.obtener_vecinos(nodo_inicio):
            if vecino not in visitados:
                # Llamada recursiva para continuar la búsqueda en los vecinos no visitados
                resultado = self.primero_profundidad(vecino, nodo_final, visitados, camino)
                if resultado:
                    return resultado
        
        # Si no se encuentra el nodo final, retroceder en el camino
        camino.pop()
        return None

    # Implementación de búsqueda en anchura (BFS)
    def primero_anchura(self, nodo_inicio, nodo_final):
        # Inicializar el conjunto de nodos visitados y la cola para la búsqueda
        visitados = set()
        cola = deque([(nodo_inicio, [nodo_inicio])])
        visitados.add(nodo_inicio)
        
        # Mientras haya nodos por explorar
        while cola:
            # Sacar el primer nodo de la cola
            nodo_actual, camino = cola.popleft()

            # Si se encuentra el nodo final, retornar el camino encontrado
            if nodo_actual == nodo_final:
                return camino
            
            # Explorar los vecinos del nodo actual
            for vecino, _ in self.obtener_vecinos(nodo_actual):
                if vecino not in visitados:
                    # Marcar el vecino como visitado y agregarlo a la cola
                    visitados.add(vecino)
                    cola.append((vecino, camino + [vecino]))

        # Si no se encuentra un camino al nodo final, retornar None
        return None
    
    # Implementación del algoritmo A*
    def a_estrella(self, nodo_inicio, nodo_final):
        # Cola de prioridad (heapq) para almacenar nodos por evaluar (costo, nodo)
        cola_prioridad = []
        heapq.heappush(cola_prioridad, (0, nodo_inicio))
        
        # Diccionario para almacenar el costo del camino más corto desde el inicio hasta un nodo
        costo_acumulado = {nodo_inicio: 0}
        
        # Diccionario para reconstruir el camino
        came_from = {nodo_inicio: None}
        
        while cola_prioridad:
            # Obtener el nodo con el menor costo total estimado (costo_acumulado + heurística)
            _, nodo_actual = heapq.heappop(cola_prioridad)

            # Si se ha alcanzado el nodo final, reconstruir y retornar el camino
            if nodo_actual == nodo_final:
                camino = []
                while nodo_actual:
                    camino.append(nodo_actual)
                    nodo_actual = came_from[nodo_actual]
                return camino[::-1]  # Retornar el camino en orden desde el inicio hasta el final
            
            # Evaluar los vecinos del nodo actual
            for vecino, costo_movimiento in self.obtener_vecinos(nodo_actual):
                nuevo_costo = costo_acumulado[nodo_actual] + costo_movimiento

                # Si este camino es el más corto conocido hasta el vecino, actualizar registros
                if vecino not in costo_acumulado or nuevo_costo < costo_acumulado[vecino]:
                    costo_acumulado[vecino] = nuevo_costo
                    prioridad = nuevo_costo + self.h(vecino, nodo_final)
                    heapq.heappush(cola_prioridad, (prioridad, vecino))
                    came_from[vecino] = nodo_actual
        
        # Si no se encuentra el camino, retornar None
        return None


# ------------------- Uso DFS y BFS ----------------------

# Leer la lista de adyacencia desde el archivo 'laberinto.txt'
lista_nodos_adyacentes, nodo_inicio, nodo_meta = leer_matriz_adyacencia('laberinto.txt')

# Crear una instancia del grafo con la lista de adyacencia
grafo = Grafo(lista_nodos_adyacentes)

# DFS
print("Primero_profundidad:")
camino_primero_profundidad = grafo.primero_profundidad(nodo_inicio, nodo_meta)
print(camino_primero_profundidad if camino_primero_profundidad else "No se encontró camino")

# BFS
print("\nPrimero_anchura:")
camino_primero_anchura = grafo.primero_anchura(nodo_inicio, nodo_meta)
print(camino_primero_anchura if camino_primero_anchura else "No se encontró camino")

# A*
print("\nAlgoritmo A*:")
a_estrella_camino = grafo.a_estrella(nodo_inicio, nodo_meta)
print(a_estrella_camino if a_estrella_camino else "No se encontró camino con algoritmo A*")