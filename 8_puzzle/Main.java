public class Main{
    public static void main(String[] args) {
        Puzzle8 puzzle = new Puzzle8();
        String[] sucesores = puzzle.generarSucesores(puzzle.initialState);
        System.out.println("Sucesores del estado inicial:");
        for (String sucesor : sucesores) {
            System.out.println(sucesor);
        }
        
    }
}