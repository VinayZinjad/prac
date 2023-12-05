public class ShapeFactory implements AbstractFactory {

    @Override
    public Shape createCircle() {
        return new Circle();
    }

    @Override
    public Shape createSquare() {
        return new Square();
    }

    @Override
    public Shape createRectangle() {
        return new Rectangle();
    }
}
