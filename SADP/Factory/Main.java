
import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);

        System.out.print("Enter shape type (Circle/Square/Rectangle): ");
        String shapeType = scanner.nextLine();

        // Use factory method to create a shape
        Shape shape = ShapeFactory.createShape(shapeType);

        // Draw the shape
        shape.draw();
    }
}