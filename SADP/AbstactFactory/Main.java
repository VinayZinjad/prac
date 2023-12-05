public class Main {

    public static void main(String[] args) {
        AbstractFactory factory = new ShapeFactory();

        Shape circle = factory.createCircle();
        circle.draw();

        Shape square = factory.createSquare();
        square.draw();

        Shape rectangle = factory.createRectangle();
        rectangle.draw();
    }
}
