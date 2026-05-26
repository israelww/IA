package diagnostico;

//Clase Nodo
class Nodo {
    String nombre; 
    Nodo izquierda;
    Nodo derecha;

    public Nodo(String nombre) {
        this.nombre = nombre;
        this.izquierda = null;
        this.derecha = null;
    }
}

// Arbol
public class Arbol {
    Nodo raiz;

    public Arbol() {
        this.raiz = null;
    }

    // vacio(): boolean
    public boolean vacio() {
        return raiz == null;
    }

    // Método: buscarNodo(nombre): Nodo
    public Nodo buscarNodo(String nombre) {
        Nodo actual = raiz;

        while (actual != null) {
            if (nombre.equals(actual.nombre)) {
                return actual; 
            } 
            else if (nombre.compareTo(actual.nombre) < 0) {
                actual = actual.izquierda;
            } 
            else {
                actual = actual.derecha;
            }
        }
        
        return null;
    }

 
    public void insertar(String nombre) {
        raiz = insertarRec(raiz, nombre);
    }

    private Nodo insertarRec(Nodo raiz, String nombre) {
        if (raiz == null) {
            raiz = new Nodo(nombre);
            return raiz;
        }
        if (nombre.compareTo(raiz.nombre) < 0)
            raiz.izquierda = insertarRec(raiz.izquierda, nombre);
        else if (nombre.compareTo(raiz.nombre) > 0)
            raiz.derecha = insertarRec(raiz.derecha, nombre);
        return raiz;
    }

 
    public static void main(String[] args) {
        Arbol miArbol = new Arbol();

        System.out.println("¿Está vacío?: " + miArbol.vacio()); // true

        miArbol.insertar("Carlos");
        miArbol.insertar("Ana");
        miArbol.insertar("Zack");

        System.out.println("¿Está vacío?: " + miArbol.vacio()); // false

        Nodo resultado = miArbol.buscarNodo("Ana");
        if (resultado != null) {
            System.out.println("Nodo encontrado: " + resultado.nombre);
        } else {
            System.out.println("Nodo no encontrado");
        }
    }
}