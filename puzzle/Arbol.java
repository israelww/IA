package puzzle;

import java.util.*;

public class Arbol {
    Nodo raiz;

    public Arbol(Nodo raiz) {
        this.raiz = raiz;
    }

    Nodo BusquedaAnchura(String estadoObjetivo) {
        if (raiz == null) return null;

        HashSet<String> visitados = new HashSet<>();
        Queue<Nodo> cola = new LinkedList<>();
        cola.add(raiz);
        visitados.add(raiz.estado);

        while (!cola.isEmpty()) {
            Nodo actual = cola.poll();
            if (actual.estado.equals(estadoObjetivo)) {
                return actual;
            }

           List<Nodo> sucesores = actual.generarSucesores();
            for (Nodo hijo : sucesores) {
                if (!visitados.contains(hijo.estado)) {
                    visitados.add(hijo.estado);
                    cola.add(hijo);
                }
            }
        }
        return null;
    }

    Nodo BusquedaProfundidad(String estadoObjetivo) {
        if (raiz == null) return null;

        HashSet<String> visitados = new HashSet<>();
        Stack<Nodo> pila = new Stack<>();
        pila.push(raiz);
        visitados.add(raiz.estado);

        while (!pila.isEmpty()) {
            Nodo actual = pila.pop();
            if (actual.estado.equals(estadoObjetivo)) {
                return actual;
            }

            List<Nodo> sucesores = actual.generarSucesores();
            for (Nodo hijo : sucesores) {
                if (!visitados.contains(hijo.estado)) {
                    visitados.add(hijo.estado);
                    pila.push(hijo);
                }
            }
        }
        return null;
    }

    Nodo BusquedaCostoUniforme(String estadoObjetivo) {
        if (raiz == null) return null;

        HashSet<String> visitados = new HashSet<>();

        // PriorityQueue ordenará los nodos automáticamente usando el compareTo que creamos
        PriorityQueue<Nodo> frontera = new PriorityQueue<>();

        frontera.add(raiz);
        while (!frontera.isEmpty()) {
            Nodo actual = frontera.poll();

            if (actual.estado.equals(estadoObjetivo)) {
                return actual;
            }
            if (!visitados.contains(actual.estado)) {
                visitados.add(actual.estado);
                List<Nodo> sucesores = actual.generarSucesores();
                for (Nodo hijo : sucesores) {
                    if (!visitados.contains(hijo.estado)) {
                        frontera.add(hijo);
                    }
                }
            }
        }
        return null;
    }

    Nodo BusquedaEsquinas(String estadoObjetivo) {

        if (raiz == null) return null;

        PriorityQueue<Nodo> abiertos = new PriorityQueue<>();
        HashSet<String> cerrados = new HashSet<>();

        raiz.costo = raiz.nivel + raiz.heuristicaEsquinas(estadoObjetivo);
        abiertos.add(raiz);

        while (!abiertos.isEmpty()) {

            Nodo actual = abiertos.poll();

            if (actual.estado.equals(estadoObjetivo))
                return actual;

            cerrados.add(actual.estado);

            for (Nodo hijo : actual.generarSucesores()) {

                if (!cerrados.contains(hijo.estado)) {

                    int g = hijo.nivel;
                    int h = hijo.heuristicaEsquinas(estadoObjetivo);

                    hijo.costo = g + h;

                    abiertos.add(hijo);
                }
            }
        }

        return null;
    }
}
