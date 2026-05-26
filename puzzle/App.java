package puzzle;

import java.util.Scanner;

public class App {

    public static void main(String[] args) {

        Scanner sc = new Scanner(System.in);

        String estadoInicial = "1238 4765";
        String objetivo = "1284376 5";

        int opcion;

        do {
            System.out.println("\n====== PUZZLE 8 ======");
            System.out.println("Estado inicial: " + estadoInicial);
            System.out.println("Estado objetivo: " + objetivo);
            System.out.println("----------------------");
            System.out.println("1. Búsqueda en Anchura ");
            System.out.println("2. Búsqueda en Profundidad ");
            System.out.println("3. Búsqueda por Costo Uniforme");
            System.out.println("4. Heurística Esquinas ");
            System.out.println("5. Tabla comparativa de todos los métodos");
            System.out.println("6. Salir");
            System.out.print("Seleccione una opción: ");

            opcion = sc.nextInt();

            if (opcion >= 1 && opcion <= 4) ejecutarMetodo(opcion, estadoInicial, objetivo);
            else if (opcion == 5) tablaComparativa(estadoInicial, objetivo);
            else if (opcion == 6) System.out.println("Saliendo del programa...");
            else System.out.println("Opción inválida.");
        } while (opcion != 6);
        sc.close();
    }

    private static void ejecutarMetodo(int opcion, String estadoInicial, String objetivo) {

        Nodo raiz = new Nodo(estadoInicial);
        Arbol puzzle = new Arbol(raiz);
        Nodo resultado = null;

        long inicio = System.currentTimeMillis();

        switch (opcion) {
            case 1:
                resultado = puzzle.BusquedaAnchura(objetivo);
                break;
            case 2:
                resultado = puzzle.BusquedaProfundidad(objetivo);
                break;
            case 3:
                resultado = puzzle.BusquedaCostoUniforme(objetivo);
                break;
            case 4:
                resultado = puzzle.BusquedaEsquinas(objetivo);
                break;
        }

        long fin = System.currentTimeMillis();

        if (resultado != null) {
            System.out.println("\n===== SOLUCIÓN ENCONTRADA =====");
            resultado.imprimirCamino();
            System.out.println("Profundidad (nivel): " + resultado.nivel);
            System.out.println("Tiempo de ejecución: " + (fin - inicio) + " ms");
        } else {
            System.out.println("No se encontró solución.");
        }
    }

    private static void tablaComparativa(String estadoInicial, String objetivo) {

        System.out.println("\n================ TABLA COMPARATIVA ================");

        String[] nombres = {
                "BFS",
                "DFS",
                "Costo Uniforme",
                "Esquinas"
        };

        for (int i = 1; i <= 4; i++) {

            Nodo raiz = new Nodo(estadoInicial);
            Arbol puzzle = new Arbol(raiz);
            Nodo resultado = null;

            long inicio = System.currentTimeMillis();

            switch (i) {
                case 1:
                    resultado = puzzle.BusquedaAnchura(objetivo);
                    break;
                case 2:
                    resultado = puzzle.BusquedaProfundidad(objetivo);
                    break;
                case 3:
                    resultado = puzzle.BusquedaCostoUniforme(objetivo);
                    break;
                case 4:
                    resultado = puzzle.BusquedaEsquinas(objetivo);
                    break;
            }

            long fin = System.currentTimeMillis();

            long tiempo = fin - inicio;

            if (resultado != null) {
                System.out.printf("%-18s | Profundidad: %-4d | Tiempo: %-5d ms\n",
                        nombres[i - 1],
                        resultado.nivel,
                        tiempo);
            } else {
                System.out.printf("%-18s | No encontró solución | Tiempo: %-5d ms\n",
                        nombres[i - 1],
                        tiempo);
            }
        }

        System.out.println("===================================================\n");
    }
}
