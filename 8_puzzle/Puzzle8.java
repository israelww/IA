public class Puzzle8 {
    String initialState = "1238 756";
    String goalState = "12 345678";

    String[] generarSucesores(String estado) {
        java.util.ArrayList<String> sucesores = new java.util.ArrayList<>();
        int indice = estado.indexOf(' ');

        // Movimiento arriba
        if (indice - 3 >= 0) {
            sucesores.add(intercambiar(estado, indice, indice - 3));
        }
        // Movimiento abajo
        if (indice + 3 < 9) {
            sucesores.add(intercambiar(estado, indice, indice + 3));
        }
        // Movimiento izquierda
        if (indice % 3 != 0) {
            sucesores.add(intercambiar(estado, indice, indice - 1));
        }
        // Movimiento derecha
        if (indice % 3 != 2) {
            sucesores.add(intercambiar(estado, indice, indice + 1));
        }

        return sucesores.toArray(new String[0]);
    }

    // Intercambia el espacio con la posición destino usando substrings
    private String intercambiar(String estado, int i, int j) {
        char[] arr = estado.toCharArray();
        char temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
        return new String(arr);
    }
}
