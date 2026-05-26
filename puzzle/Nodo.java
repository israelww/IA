package puzzle;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedList;

public class Nodo implements Comparable<Nodo> {

    String estado;
    int nivel;   // g(n)
    int costo;   // f(n) = g(n) + h(n)
    Nodo padre;

    public Nodo(String estado) {
        this.estado = estado;
        this.nivel = 0;
        this.costo = 0;
        this.padre = null;
    }

    public Nodo(String estado, int nivel, Nodo padre) {
        this.estado = estado;
        this.nivel = nivel;
        this.costo = 0;
        this.padre = padre;
    }

    @Override
    public int compareTo(Nodo otro) {
        return Integer.compare(this.costo, otro.costo);
    }

    public LinkedList<Nodo> generarSucesores() {
        LinkedList<Nodo> sucesores = new LinkedList<>();

        int indice = this.estado.indexOf(' ');
        int nuevoNivel = this.nivel + 1;

        switch (indice) {

            case 0:
                sucesores.add(new Nodo(intercambiar(0,1), nuevoNivel, this));
                sucesores.add(new Nodo(intercambiar(0,3), nuevoNivel, this));
                break;

            case 1:
                sucesores.add(new Nodo(intercambiar(1,0), nuevoNivel, this));
                sucesores.add(new Nodo(intercambiar(1,2), nuevoNivel, this));
                sucesores.add(new Nodo(intercambiar(1,4), nuevoNivel, this));
                break;

            case 2:
                sucesores.add(new Nodo(intercambiar(2,1), nuevoNivel, this));
                sucesores.add(new Nodo(intercambiar(2,5), nuevoNivel, this));
                break;

            case 3:
                sucesores.add(new Nodo(intercambiar(3,0), nuevoNivel, this));
                sucesores.add(new Nodo(intercambiar(3,4), nuevoNivel, this));
                sucesores.add(new Nodo(intercambiar(3,6), nuevoNivel, this));
                break;

            case 4:
                sucesores.add(new Nodo(intercambiar(4,1), nuevoNivel, this));
                sucesores.add(new Nodo(intercambiar(4,3), nuevoNivel, this));
                sucesores.add(new Nodo(intercambiar(4,5), nuevoNivel, this));
                sucesores.add(new Nodo(intercambiar(4,7), nuevoNivel, this));
                break;

            case 5:
                sucesores.add(new Nodo(intercambiar(5,2), nuevoNivel, this));
                sucesores.add(new Nodo(intercambiar(5,4), nuevoNivel, this));
                sucesores.add(new Nodo(intercambiar(5,8), nuevoNivel, this));
                break;

            case 6:
                sucesores.add(new Nodo(intercambiar(6,3), nuevoNivel, this));
                sucesores.add(new Nodo(intercambiar(6,7), nuevoNivel, this));
                break;

            case 7:
                sucesores.add(new Nodo(intercambiar(7,4), nuevoNivel, this));
                sucesores.add(new Nodo(intercambiar(7,6), nuevoNivel, this));
                sucesores.add(new Nodo(intercambiar(7,8), nuevoNivel, this));
                break;

            case 8:
                sucesores.add(new Nodo(intercambiar(8,5), nuevoNivel, this));
                sucesores.add(new Nodo(intercambiar(8,7), nuevoNivel, this));
                break;
        }

        return sucesores;
    }

    private String intercambiar(int i, int j) {
        StringBuilder sb = new StringBuilder(this.estado);
        char temp = sb.charAt(i);
        sb.setCharAt(i, sb.charAt(j));
        sb.setCharAt(j, temp);
        return sb.toString();
    }

    public int heuristicaEsquinas(String estadoObjetivo) {

        int h = 0;

        int[] esquinas = {0, 2, 6, 8};

        for (int esquina : esquinas) {
            char actual = this.estado.charAt(esquina);
            char objetivo = estadoObjetivo.charAt(esquina);

            if (actual != ' ' && actual != objetivo) {
                h++;
            }
        }

        return h;
    }

    public void imprimirCamino() {

        ArrayList<Nodo> camino = new ArrayList<>();
        Nodo actual = this;

        while (actual != null) {
            camino.add(actual);
            actual = actual.padre;
        }

        Collections.reverse(camino);

        for (Nodo n : camino) {

            for (int i = 0; i < 9; i++) {
                System.out.print("[" + n.estado.charAt(i) + "] ");
                if ((i + 1) % 3 == 0) System.out.println();
            }

            System.out.println("Nivel g(n): " + n.nivel +
                    " | f(n): " + n.costo);
            System.out.println("----------------");
        }
    }
}
