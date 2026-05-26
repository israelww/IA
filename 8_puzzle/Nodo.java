public class Nodo {
    private String estado;
    private Nodo padre;
    int nivel;
    int costo;



    public Nodo(String estado, Nodo padre) {
        this.estado = estado;
        this.padre = padre;
        
    }
    public String getEstado() {
        return estado;
    }
    public Nodo getPadre() {
        return padre;
    }

}
